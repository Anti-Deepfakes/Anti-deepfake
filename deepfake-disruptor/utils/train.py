import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import transforms
from model.U_Net import UNet
import insightface
import cv2
from insightface.app import FaceAnalysis
from utils.evaluation import validate
import numpy as np

def tensor_to_cv2_image(tensor_image):
    image_np = tensor_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
    
    image_np = image_np.astype(np.uint8)
    
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2


def criterion_identity(predicted, target):
    return 1- torch.sigmoid(nn.MSELoss()(predicted, target))


def hinge_loss(perturbation, epsilon):
    perturbation_norm = torch.norm(torch.ones(perturbation.shape).float().cuda() - perturbation, p=float('inf'), dim=(1, 2, 3))
    loss = torch.mean(perturbation_norm)
    return loss


def gradnorm_update(losses, initial_losses, alpha, weights):
    avg_loss = sum(losses) / len(losses)
    loss_ratios = [loss / initial_loss for loss, initial_loss in zip(losses, initial_losses)]
    relative_losses = [ratio / avg_loss for ratio in loss_ratios]
    grad_norms = [(relative_loss ** alpha) * weight for relative_loss, weight in zip(relative_losses, weights)]
    new_weights = [weight * (grad_norm / sum(grad_norms)) for weight, grad_norm in zip(weights, grad_norms)]
    return new_weights


def iou_loss(pred_boxes, true_boxes):
    inter_area = torch.max(torch.min(pred_boxes[..., 2], true_boxes[..., 2]) - torch.max(pred_boxes[..., 0], true_boxes[..., 0]), torch.tensor(0.0))
    inter_area *= torch.max(torch.min(pred_boxes[..., 3], true_boxes[..., 3]) - torch.max(pred_boxes[..., 1], true_boxes[..., 1]), torch.tensor(0.0))

    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])

    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area

    return 1 - torch.mean(iou)


def train(hp, train_loader, valid_loader, chkpt_path, save_dir):
    run = wandb.init(
        project=hp.log.project_name,
        config={
            "learning_rate": hp.train.lr,
            "architecture": "U-Net",
            "dataset": "celeba",
            "epochs": hp.train.epochs,
        }
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", device)
    init_epoch = 0
    perturbation_generator = UNet(3).to(torch.device("cuda:0")).train()
    
    face_detector = FaceAnalysis(name='buffalo_l')
    face_detector.prepare(ctx_id=0, det_size=(hp.data.image_size, hp.data.image_size))
    
    perturbation_generator = nn.DataParallel(perturbation_generator, device_ids=[0, 1, 2])
    # face_detector = nn.DataParallel(face_detector, device_ids=[0, 1, 2])

    optimizer = optim.Adam(perturbation_generator.parameters(), lr=hp.train.lr)

    initial_losses = None
    if chkpt_path is not None:
        print(chkpt_path)
        checkpoint = torch.load(chkpt_path)
        perturbation_generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']

    wandb.watch(perturbation_generator)
    
    for epoch in range(init_epoch, hp.train.epochs):
        for real_images, image_weighted, truth_bbox, truth_landmarks in train_loader:
            # validate(valid_loader, perturbation_generator, face_detector, device, hp)

            real_images = real_images.to(torch.device("cuda:0"))
            image_weighted = image_weighted.to(device)
            truth_bbox = truth_bbox.to(device)
            truth_landmarks = truth_landmarks.to(device)

            perturbations = perturbation_generator(image_weighted)
            perturbed_images = real_images * perturbations
            # perturbed_images = torch.clamp(perturbed_images, 0, 1)

            pred_bbox = []
            pred_landmarks = []
            for idx, img in enumerate(perturbed_images):
                num_image = tensor_to_cv2_image(img.clone().detach())

                try:
                    train_face = face_detector.get(num_image)
                    train_face = train_face[0]
                    pred_bbox.append(torch.tensor(train_face['bbox']).unsqueeze(0).float().to(device))
                    pred_landmarks.append(torch.tensor(train_face['landmark_3d_68']).unsqueeze(0).float().to(device))
                    
                except Exception as e:
                    print(e)
                    pred_bbox.append(torch.zeros([1, 4]).float().to(device))
                    pred_landmarks.append(torch.zeros([1, 68, 3]).float().to(device))

            pred_bbox = torch.cat(pred_bbox, dim=0).to(device)
            pred_landmarks = torch.cat(pred_landmarks, dim=0).to(device)

            perturbation_loss = hinge_loss(perturbations, hp.train.epsilon)
            identity_loss = 1.0 / (1.0 + nn.MSELoss()(perturbed_images, real_images))
            
            bbox_loss = iou_loss(pred_bbox, truth_bbox)
            landmarks_loss = 1.0 / (1.0 + nn.MSELoss()(pred_landmarks, truth_landmarks))

            # losses = [bbox_loss, landmarks_loss, perturbation_loss, identity_loss]
            
            # print(bbox_loss)
            # print(landmarks_loss)

            total_loss = (
                    bbox_loss + landmarks_loss + 0.5*perturbation_loss + identity_loss
            )
            
            # print(total_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print('epoch : ', epoch, 'loss : ', total_loss.item(),
                    'bbox_loss : ', bbox_loss.item(),
                    'landmarks_loss : ', landmarks_loss.item(),
                  'perturbation_loss : ', perturbation_loss.item(),
                  'identity_loss : ', identity_loss.item())
            wandb.log({
                'epoch': epoch,
                'loss': total_loss.item(),
                'bbox_loss' : bbox_loss.item(),
                'landmarks_loss' : landmarks_loss.item(),
                'perturbation_loss': perturbation_loss.item(),
                'identity_loss': identity_loss.item(),
            })

            checkpoint_path = f"{save_dir}/unet_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': perturbation_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item()
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            print(f"Epoch [{epoch + 1}/{hp.train.epochs}], Loss: {total_loss.item()}")
        validate(valid_loader, perturbation_generator, face_detector, device, hp)
