import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import transforms
from app.model.U_Net import UNet
import cv2
from insightface.app import FaceAnalysis
from app.utils.evaluation import validate
import os
import numpy as np
import mlflow


def tensor_to_cv2_image(tensor_image):
    print("[LOG: tensor_to_cv2_image] Converting tensor to OpenCV image format.")
    image_np = tensor_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2


def criterion_identity(predicted, target):
    print("[LOG: criterion_identity] Calculating identity loss.")
    return 1 - torch.sigmoid(nn.MSELoss()(predicted, target))


def hinge_loss(perturbation, epsilon):
    print(f"[LOG: hinge_loss] Calculating hinge loss with epsilon: {epsilon}.")
    perturbation_norm = torch.norm(torch.ones(perturbation.shape).float().cuda() - perturbation, p=float('inf'), dim=(1, 2, 3))
    loss = torch.mean(perturbation_norm)
    return loss


def iou_loss(pred_boxes, true_boxes):
    print("[LOG: iou_loss] Calculating IOU loss.")
    inter_area = torch.max(torch.min(pred_boxes[..., 2], true_boxes[..., 2]) - torch.max(pred_boxes[..., 0], true_boxes[..., 0]), torch.tensor(0.0))
    inter_area *= torch.max(torch.min(pred_boxes[..., 3], true_boxes[..., 3]) - torch.max(pred_boxes[..., 1], true_boxes[..., 1]), torch.tensor(0.0))

    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])

    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area

    return 1 - torch.mean(iou)


def train(hp, train_loader, valid_loader, chkpt_path, save_dir, db, version, data_version):
    print("[LOG: train] Starting training process.")
    print(f"[LOG: train] Hyperparameters: {hp}")
    print(f"[LOG: train] Checkpoint path: {chkpt_path if chkpt_path else 'None'}")
    print(f"[LOG: train] Save directory: {save_dir}")

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
    print(f"[LOG: train] Device: {device}")

    init_epoch = 0
    print("[LOG: train] Initializing U-Net model.")
    perturbation_generator = UNet(3).to(device).train()
    
    try:
        print("[DEBUG: train] Initializing FaceAnalysis object.")
        face_detector = FaceAnalysis(name='buffalo_l')
        print("[DEBUG: train] FaceAnalysis object initialized successfully.")
    except Exception as e:
        print(f"[ERROR: train] FaceAnalysis initialization failed: {str(e)}")
        raise

    print("[LOG: FaceAnalysis] Initializing FaceAnalysis with ctx_id=7.")

    try:
        face_detector.prepare(ctx_id=7, det_size=(hp.data.image_size, hp.data.image_size))
        print("[LOG: FaceAnalysis] FaceAnalysis prepared successfully.")
    except Exception as e:
        print(f"[ERROR: FaceAnalysis] Failed during preparation: {e}")
        raise
    print("[LOG: train] FaceAnalysis initialized successfully.")

    print("[LOG: train] Setting up DataParallel for multi-GPU training.")
    # perturbation_generator = nn.DataParallel(perturbation_generator, device_ids=[0, 1, 2])
    perturbation_generator = nn.DataParallel(perturbation_generator, device_ids=[0])

    print("[LOG: train] Setting up optimizer.")
    optimizer = optim.Adam(perturbation_generator.parameters(), lr=hp.train.lr)

    if chkpt_path is not None:
        print(f"[LOG: train] Loading checkpoint from {chkpt_path}.")
        checkpoint = torch.load(chkpt_path)
        perturbation_generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        print(f"[LOG: train] Resuming from epoch {init_epoch}.")

    wandb.watch(perturbation_generator)

    # train loader 확인
    print(f"[DEBUG] train_loader has {len(train_loader)} batches.")
    for batch_idx, batch in enumerate(train_loader):
        print(f"[DEBUG] Batch {batch_idx + 1}: {batch}")

    # Training loop
    for epoch in range(init_epoch, hp.train.epochs):
        print(f"[LOG: train] Starting epoch {epoch + 1}/{hp.train.epochs}.")
        if (epoch+1) == hp.train.epochs: 
            print("escape from epoch")
            break
        for batch_idx, (real_images, image_weighted, truth_bbox, truth_landmarks) in enumerate(train_loader):
            print(f"[DEBUG: train] Processing batch {batch_idx + 1}.")
            try:
                # 출력: 데이터의 크기 및 내용 확인
                print(f"[DEBUG: train] real_images shape: {real_images.shape}, dtype: {real_images.dtype}")
                print(f"[DEBUG: train] image_weighted shape: {image_weighted.shape}, dtype: {image_weighted.dtype}")
                print(f"[DEBUG: train] truth_bbox shape: {truth_bbox.shape}, dtype: {truth_bbox.dtype}")
                print(f"[DEBUG: train] truth_landmarks shape: {truth_landmarks.shape}, dtype: {truth_landmarks.dtype}")

                real_images = real_images.to(device)
                image_weighted = image_weighted.to(device)
                truth_bbox = truth_bbox.to(device)
                truth_landmarks = truth_landmarks.to(device)

                print(f"[DEBUG: train] real_images on {device} shape: {real_images.shape}, dtype: {real_images.dtype}")
                print(f"[DEBUG: train] image_weighted on {device} shape: {image_weighted.shape}, dtype: {image_weighted.dtype}")
                print(f"[DEBUG: train] truth_bbox on {device} shape: {truth_bbox.shape}, dtype: {truth_bbox.dtype}")
                print(f"[DEBUG: train] truth_landmarks on {device} shape: {truth_landmarks.shape}, dtype: {truth_landmarks.dtype}")

                perturbations = perturbation_generator(image_weighted)
                perturbed_images = real_images * perturbations

                pred_bbox = []
                pred_landmarks = []
                for idx, img in enumerate(perturbed_images):
                    print(f"[DEBUG: train] Processing image {idx + 1} in batch {batch_idx + 1}.")
                    num_image = tensor_to_cv2_image(img.clone().detach())
                    try:
                        train_face = face_detector.get(num_image)
                        if train_face:
                            train_face = train_face[0]
                            pred_bbox.append(torch.tensor(train_face['bbox']).unsqueeze(0).float().to(device))
                            pred_landmarks.append(torch.tensor(train_face['landmark_3d_68']).unsqueeze(0).float().to(device))
                            print(f"[DEBUG: train] Face detected for image {idx + 1}.")
                        else:
                            print(f"[WARN: train] No face detected for image {idx + 1}.")
                            pred_bbox.append(torch.zeros([1, 4]).float().to(device))
                            pred_landmarks.append(torch.zeros([1, 68, 3]).float().to(device))
                    except Exception as e:
                        print(f"[ERROR: train] Face detection failed for image {idx + 1}: {e}")
                        pred_bbox.append(torch.zeros([1, 4]).float().to(device))
                        pred_landmarks.append(torch.zeros([1, 68, 3]).float().to(device))

                pred_bbox = torch.cat(pred_bbox, dim=0)
                pred_landmarks = torch.cat(pred_landmarks, dim=0)

                perturbation_loss = hinge_loss(perturbations, hp.train.epsilon)
                identity_loss = criterion_identity(perturbed_images, real_images)
                bbox_loss = iou_loss(pred_bbox, truth_bbox)
                landmarks_loss = 1.0 / (1.0 + nn.MSELoss()(pred_landmarks, truth_landmarks))

                total_loss = bbox_loss + landmarks_loss + 0.5 * perturbation_loss + identity_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                print(f"[LOG: train] Batch {batch_idx + 1}, Loss: {total_loss.item()}, "
                        f"BBox Loss: {bbox_loss.item()}, Landmarks Loss: {landmarks_loss.item()}, "
                        f"Perturbation Loss: {perturbation_loss.item()}, Identity Loss: {identity_loss.item()}")

                wandb.log({
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'loss': total_loss.item(),
                    'bbox_loss': bbox_loss.item(),
                    'landmarks_loss': landmarks_loss.item(),
                    'perturbation_loss': perturbation_loss.item(),
                    'identity_loss': identity_loss.item(),
                })

            except Exception as e:
                print(f"[ERROR: train] Exception in batch {batch_idx + 1}: {e}")
                raise

        # Save checkpoint
        checkpoint_path = f"{save_dir}/unet_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': perturbation_generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss.item()
        }, checkpoint_path)
        print(f"[LOG: train] Checkpoint saved at {checkpoint_path}.")

    validate(
        dataloader=valid_loader,
        perturbation_generator=perturbation_generator,
        face_detector=face_detector,
        device=device,
        hp=hp,
        db=db,
        version=version,
        data_version=data_version,
    )

    # 중복저장? 일단 보자
    final_model_path = os.path.join(save_dir, f"model_{data_version}.pth")
    torch.save(perturbation_generator.state_dict(), final_model_path)
    print(f"[LOG: train] Final model saved at {final_model_path}.")

    mlflow.pytorch.log_model(
        pytorch_model=perturbation_generator,
        artifact_path="/mlflow/model"
    )
    print("[LOG: train] Model logged to MLflow.")
