import torch
import torch.nn as nn
import wandb
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from io import BytesIO
from sklearn.metrics import precision_score, recall_score
import cv2
import numpy as np
from sqlalchemy.orm import Session
from app.db_models import Performance

# Validation function to evaluate model on validation set
def validate(dataloader, perturbation_generator, face_detector, device, hp, db, version, data_version):
    print("[LOG: validate] Starting validation process.")
    print(f"  - Version: {version}")
    print(f"  - Data version: {data_version}")
    perturbation_generator.eval()
    all_bbox_loss = 0
    all_landmarks_loss = 0
    all_perturbation_loss = 0
    all_identity_loss = 0
    all_total_loss = 0
    counts = 0
    images = []
    with torch.no_grad():
        for i, (real_images, image_weighted, truth_bbox, truth_landmarks) in enumerate(dataloader):
            print(f"[LOG: validate] Processing batch {i + 1}.")
            counts += 1
            real_images = real_images.to(device)
            image_weighted = image_weighted.to(device)
            truth_bbox = truth_bbox.to(device)
            truth_landmarks = truth_landmarks.to(device)

            perturbations = perturbation_generator(image_weighted)
            perturbed_images = real_images * perturbations

            num_image = tensor_to_cv2_image(perturbed_images)
            images.append(num_image)
            try:
                train_face = face_detector.get(num_image)
                train_face = train_face[0]
                pred_bbox = torch.tensor(train_face['bbox']).unsqueeze(0).float().to(device)
                pred_landmarks = torch.tensor(train_face['landmarks']).unsqueeze(0).float().to(device)
                print(f"[LOG: validate] Face detected successfully for batch {i + 1}.")
            except Exception as e:
                print(f"[ERROR: validate] Face detection failed for batch {i + 1}: {str(e)}")
                pred_bbox = torch.zeros([4]).unsqueeze(0).float().to(device)
                pred_landmarks = torch.zeros([68, 3]).unsqueeze(0).float().to(device)

            perturbation_loss = hinge_loss(perturbations, hp.train.epsilon)
            identity_loss = 1.0 / (1.0 + nn.MSELoss()(perturbed_images, real_images))
            bbox_loss = iou_loss(pred_bbox, truth_bbox)
            landmarks_loss = 1.0 / (1.0 + nn.MSELoss()(pred_landmarks, truth_landmarks))

            total_loss = bbox_loss + landmarks_loss + perturbation_loss + identity_loss

            all_perturbation_loss += perturbation_loss.item()
            all_bbox_loss += bbox_loss.item()
            all_landmarks_loss += landmarks_loss.item()
            all_identity_loss += identity_loss.item()
            all_total_loss = total_loss.item()

            print(f"[LOG: validate] Losses for batch {i + 1}:")
            print(f"  - bbox_loss: {bbox_loss.item()}")
            print(f"  - landmarks_loss: {landmarks_loss.item()}")
            print(f"  - perturbation_loss: {perturbation_loss.item()}")
            print(f"  - identity_loss: {identity_loss.item()}")

    all_perturbation_loss /= counts
    all_bbox_loss /= counts
    all_landmarks_loss /= counts
    all_identity_loss /= counts

    perturbation_generator.train()

    print("[LOG: validate] Logging validation metrics to WandB.")
    wandb.log({
        'val_loss': all_total_loss,
        'val_bbox_loss': all_bbox_loss,
        'val_landmarks_loss': all_landmarks_loss,
        'val_perturbation_loss': all_perturbation_loss,
        'val_identity_loss': all_identity_loss,
        "images": [wandb.Image(img, caption=f"Image {i + 1}") for i, img in enumerate(images)]
    })

    print("[LOG: validate] Saving performance metrics to DB.")
    save_performance_to_db(
        db=db,
        model_type=1,
        version=version,
        data_version=data_version,
        bbox_loss=all_bbox_loss,
        landmarks_loss=all_landmarks_loss,
        perturbation_loss=all_perturbation_loss,
        identity_loss=all_identity_loss,
        total_loss=all_total_loss
    )
    print("[LOG: validate] Validation process completed successfully.")

def iou_loss(pred_boxes, true_boxes):
    print("[LOG: iou_loss] Calculating IoU loss.")
    inter_area = torch.max(torch.min(pred_boxes[..., 2], true_boxes[..., 2]) - torch.max(pred_boxes[..., 0], true_boxes[..., 0]), torch.tensor(0.0))
    inter_area *= torch.max(torch.min(pred_boxes[..., 3], true_boxes[..., 3]) - torch.max(pred_boxes[..., 1], true_boxes[..., 1]), torch.tensor(0.0))

    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])

    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area
    loss = 1 - torch.mean(iou)
    print(f"[LOG: iou_loss] IoU loss calculated: {loss.item()}")
    return loss

def tensor_to_cv2_image(tensor_image):
    print("[LOG: tensor_to_cv2_image] Converting tensor to CV2 image.")
    image_np = tensor_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    print("[LOG: tensor_to_cv2_image] Conversion successful.")
    return image_cv2

def hinge_loss(perturbation, epsilon):
    print("[LOG: hinge_loss] Calculating hinge loss.")
    perturbation_norm = torch.norm(torch.ones(perturbation.shape).float().cuda() - perturbation, p=float('inf'), dim=(1, 2, 3))
    loss = torch.mean(perturbation_norm)
    print(f"[LOG: hinge_loss] Hinge loss calculated: {loss.item()}")
    return loss

def save_performance_to_db(
    db,
    model_type,
    version,
    data_version,
    bbox_loss,
    landmarks_loss,
    perturbation_loss,
    identity_loss,
    total_loss
):
    """
    성능 평가 결과를 DB에 저장.
    """
    print(f"[LOG: save_performance_to_db] Saving performance to DB for model version {version}.")
    try:
        performance = Performance(
            model_type=model_type,
            version=version,
            data_version=data_version,
            bbox_loss=bbox_loss,
            landmarks_loss=landmarks_loss,
            perturbation_loss=perturbation_loss,
            identity_loss=identity_loss,
            total_loss=total_loss,
        )
        db.add(performance)
        db.commit()
        print(f"[LOG: save_performance_to_db] Performance saved successfully for version {version}.")
    except Exception as e:
        print(f"[ERROR: save_performance_to_db] Failed to save performance: {str(e)}")
        db.rollback()
        raise
