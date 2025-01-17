import torch
import torch.nn as nn
import wandb
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from io import BytesIO
from sklearn.metrics import precision_score, recall_score
import cv2
import numpy as np

# Validation function to evaluate model on validation set
def validate(dataloader, perturbation_generator, face_detector, device, hp):
    perturbation_generator.eval()
    all_bbox_loss = 0
    all_landmarks_loss = 0
    all_perturbation_loss = 0
    all_identity_loss = 0
    all_total_loss = 0
    counts = 0
    images = []
    with torch.no_grad():
        for real_images, image_weighted, truth_bbox, truth_landmarks in dataloader:
            counts+=1
            real_images = real_images.to(device)
            image_weighted = image_weighted.to(device)
            truth_bbox = truth_bbox.to(device)
            truth_landmarks = truth_landmarks.to(device)

            perturbations = perturbation_generator(image_weighted)
            perturbed_images = real_images * perturbations
            # print(perturbed_images.shape)
            num_image = tensor_to_cv2_image(perturbed_images)
            images.append(num_image)
            try:
                train_face = face_detector.get(num_image)
                train_face = train_face[0]
                pred_bbox = torch.tensor(train_face['bbox']).unsqueeze(0).float().to(device)
                pred_landmarks = torch.tensor(train_face['landmarks']).unsqueeze(0).float().to(device)

            except Exception as e:
                print(e)
                pred_bbox = torch.zeros([4]).unsqueeze(0).float().to(device)
                pred_landmarks = torch.zeros([68, 3]).unsqueeze(0).float().to(device)

            perturbation_loss = hinge_loss(perturbations, hp.train.epsilon)
            identity_loss = 1.0 / (1.0 + nn.MSELoss()(perturbed_images, real_images))
            # print(pred_landmarks.shape)
            # print(truth_landmarks.shape)
            bbox_loss = iou_loss(pred_bbox, truth_bbox)
            landmarks_loss = 1.0 / (1.0 + nn.MSELoss()(pred_landmarks, truth_landmarks))

            losses = [bbox_loss, landmarks_loss, perturbation_loss, identity_loss]

            total_loss = (
                    bbox_loss + landmarks_loss + perturbation_loss + identity_loss
            )

            all_perturbation_loss+=perturbation_loss.item()
            all_bbox_loss += bbox_loss.item()
            all_landmarks_loss += landmarks_loss.item()
            all_identity_loss += identity_loss.item()
            all_total_loss = total_loss.item()

    all_perturbation_loss /= counts
    all_bbox_loss /= counts
    all_landmarks_loss /= counts
    all_identity_loss /= counts

    perturbation_generator.train()

    wandb.log({
        'val_loss': all_total_loss,
        'val_bbox_loss' : all_bbox_loss.item(),
        'val_landmarks_loss' : all_landmarks_loss.item(),
        'val_perturbation_loss': all_perturbation_loss.item(),
        'val_identity_loss': all_identity_loss.item(),
        "images": [wandb.Image(img, caption=f"Image {i+1}") for i, img in enumerate(images)]
    })

def iou_loss(pred_boxes, true_boxes):
    inter_area = torch.max(torch.min(pred_boxes[..., 2], true_boxes[..., 2]) - torch.max(pred_boxes[..., 0], true_boxes[..., 0]), torch.tensor(0.0))
    inter_area *= torch.max(torch.min(pred_boxes[..., 3], true_boxes[..., 3]) - torch.max(pred_boxes[..., 1], true_boxes[..., 1]), torch.tensor(0.0))

    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])

    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area

    return 1 - torch.mean(iou)


def tensor_to_cv2_image(tensor_image):
    image_np = tensor_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0

    image_np = image_np.astype(np.uint8)

    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2


def hinge_loss(perturbation, epsilon):
    # perturbation_norm = torch.norm(perturbation, p=float('inf'), dim=(1, 2, 3))
    
    # loss = torch.mean(torch.clamp(perturbation_norm - epsilon, min=0))
    
    # return loss
    perturbation_norm = torch.norm(torch.ones(perturbation.shape).float().cuda() - perturbation, p=float('inf'), dim=(1, 2, 3))
    loss = torch.mean(perturbation_norm)
    return loss
