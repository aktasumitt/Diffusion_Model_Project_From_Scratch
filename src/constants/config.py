class Config:
    # Data Ingestion Stage
    train_data_path = "datasets/train"

    # Data Transformation Stage
    train_dataset_save_path = "artifacts/data_transformation/train_dataset.pth"

    # Model Ingestion Stage
    u_net_save_path = "artifacts/model/Unet_model.pth"
    diffusion_save_path = "artifacts/model/difussion.pth"

    # Training Stage
    checkpoint_save_path = "callbacks/checkpoints/checkpoint_latest.pth.tar"
    final_unet_model_path = "callbacks/final_model/Unet_model.pth"
    final_diffusion_path = "callbacks/final_model/diffusion.pth"
    results_save_path = "results/train_results.json"

    # Testing Stage
    test_img_save_path = "results/test_images.jpg"

    # # Prediction Stage
    # predicted_img_save_path= "prediction_images/"
