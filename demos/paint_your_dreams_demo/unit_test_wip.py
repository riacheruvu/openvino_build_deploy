import unittest
from main import load_pipeline, get_available_devices, generate_images
from PIL import Image
import numpy as np
from pathlib import Path
import os

class TestPipeline(unittest.TestCase):

    def test_available_devices(self):
        # Test to ensure there are devices available for inference.
        devices = get_available_devices()
        self.assertTrue(len(devices) > 0, "No devices available for inference")
        print("Success: Available devices found. Full list:", devices)

    def test_load_pipeline(self):
        # Test if the pipeline loads successfully on a specific device.
        model_name = "OpenVINO/LCM_Dreamshaper_v7-fp16-ov"  # Default model name
        device = "GPU"  # Use CPU as the device for testing
        loaded_device = load_pipeline(model_name, device)
        self.assertEqual(loaded_device, device, "Failed to load pipeline to the specified device")
        print("Success: Pipeline loaded successfully on device:", device)

    def test_generate_images(self):
        # Test the generation of images from the pipeline using default parameters.
        prompt = "A sail boat on a grass field with mountains in the morning and sunny day"  # Default prompt
        seed = 0  # Default seed value
        size = 512  # Default image size
        guidance_scale = 8.0  # Default guidance scale for base
        num_inference_steps = 5  # Default number of inference steps
        randomize_seed = True  # Default randomize seed value
        device = "GPU"  # Default inference device
        endless_generation = False  # Default endless generation
        model_name = "OpenVINO/LCM_Dreamshaper_v7-fp16-ov"  # Set a valid model name

        # Load the pipeline before generating images
        load_pipeline(model_name, device)

        # Generator function call, should yield images and processing time
        image_generator = generate_images(
            prompt, seed, size, guidance_scale, num_inference_steps, randomize_seed, device, endless_generation
        )

        try:
            result_image, processing_time = next(image_generator)
            self.assertIsNotNone(result_image, "Image generation failed - No image returned.")
            self.assertIsInstance(processing_time, float, "Image generation failed - Processing time not a float.")

            # Define the directory and find an available filename incrementally
            images_dir = Path("generated_images")
            images_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist

            # Find the next available filename (e.g., test_image_1.png, test_image_2.png, etc.)
            image_index = 1
            while (images_dir / f"test_image_{image_index}.png").exists():
                image_index += 1

            result_image_path = images_dir / f"test_image_{image_index}.png"
            
            # Save the generated image for inspection
            result_image_pil = Image.fromarray(np.array(result_image))
            result_image_pil.save(result_image_path)

            # Verify that the image was saved successfully
            self.assertTrue(os.path.exists(result_image_path), f"Image was not saved at {result_image_path}")
            print("Success: Image generated successfully with processing time:", processing_time)
            print("Image saved at:", result_image_path)
        except StopIteration:
            self.fail("Image generation failed - Generator function stopped unexpectedly.")

if __name__ == '__main__':
    unittest.main()
