# DesignGenie - Interior Design AI Copilot

DesignGenie is an AI-powered Gradio app for home interior design. It acts as a language-guided copilot, utilizing language-driven segmentation and image generation techniques to provide personalized interior design suggestions for different sections of your home. The app combines the power of ControlNet and StableDiffusion models to deliver accurate and aesthetically pleasing designs.

## Features

- **Image Segmentation:** DesignGenie uses a segmentation model to identify various sections of your home in an input image. It allows you to select specific segments for redesign by simply clicking on the image.

- **Prompt-based Design Generation:** You can provide a text prompt to describe how you want the model to redesign the selected segments. DesignGenie generates multiple design outputs based on the prompt, giving you a range of design options to choose from.

- **Model Selection:** The app offers a selection of segmentation models and ControlNet modules for you to choose from. You can experiment with different models to achieve the desired results.

- **Stable Diffusion Parameters:** DesignGenie allows you to customize the Stable Diffusion parameters, including the number of inference steps, strength, guidance scale, and eta. These parameters influence the style and quality of the generated designs.

## Installation

To run DesignGenie locally, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/naderasadi/DesignGenie.git
   ```

2. Install the required dependencies:

    Using conda
    ```
    conda env create -f environment.yml
    ```

    Using pip
    ```
    pip install -r requirements.txt
    ```

3. Run the Gradio app:
    ```
    python app.py
    ```


## Models 
DesignGenie leverages the following models and techniques:

- **Segmentation Model**: The app uses a segmentation model (such as "mask2former" or "maskformer") to segment the input image and identify different sections of the home interior.

- **ControlNet**: DesignGenie utilizes the ControlNet model to generate control images that guide the design generation process. Different ControlNet modules (such as "mlsd," "soft_edge," "hed," or "scribble") can be selected to influence the style and aesthetics of the designs.

- **StableDiffusion**: The StableDiffusion technique is applied to generate multiple design outputs based on the selected sections and the provided prompts. The Stable Diffusion parameters, including the number of inference steps, strength, guidance scale, and eta, can be adjusted to control the design generation process.


