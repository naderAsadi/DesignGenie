from src.designgenie import InpaintPipeline

pipe = InpaintPipeline(
    segmentation_model_name="mask2former",
    diffusion_model_name="controlnet_inpaint",
    control_model_name="mlsd",
    images_root="/home/nader/Projects/DesignGenie/assets/images/",
    prompts_path="/home/nader/Projects/DesignGenie/assets/prompts.txt",
    image_size=(768, 512),
    image_extensions=(".jpg", ".jpeg", ".png", ".webp"),
)

pipe.run()