from src.designgenie.interfaces import GradioApp


# def run_pipeline():
#     pipe = InpaintPipeline(
#         segmentation_model_name="mask2former",
#         diffusion_model_name="controlnet_inpaint",
#         control_model_name="mlsd",
#         images_root="/home/nader/Projects/DesignGenie/assets/images/",
#         prompts_path="/home/nader/Projects/DesignGenie/assets/prompts.txt",
#         image_size=(768, 512),
#         image_extensions=(".jpg", ".jpeg", ".png", ".webp"),
#     )

#     results = pipe.run()

#     for i, images in enumerate(results):
#         for j, image in enumerate(images):
#             image.save(f"./assets/results/result_{i}_{j}.png")

if __name__ == "__main__":
    app = GradioApp()
    app.interface.launch(share=True)
