import unittest
from pathlib import Path
import numpy as np
#Define main functions in a .py file
from main import convert, get_model, run

#Phase 1: Test model conversion and that model path exists
class TestModelConversion(unittest.TestCase):
    def test_model_conversion(self):
        model_name = "yolo11n"  # Replace with your model name
        model_dir = Path("model/")  # Replace with your model directory path
        
        ov_model_path, ov_int8_model_path = convert(model_name, model_dir)
        
        self.assertTrue(ov_model_path.exists(), "FP16 model conversion failed")
        self.assertTrue(ov_int8_model_path.exists(), "INT8 model conversion failed")

#Phase 2: Test simple model input
class TestModelRunning(unittest.TestCase):
    def test_model_running(self):
        model_path = Path("/path/to/your/ov_model.xml")  # Replace with your model path
        model = get_model(model_path)
        
        input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)  # Example input data
        results = model(input_data)
        
        self.assertIsNotNone(results, "Model inference failed")

#Phase 3: Test end-to-end demo or ref kit
class TestDemo(unittest.TestCase):
    def test_end_to_end_demo(self):
        video_path = "path/to/your/video.mp4"  # Replace with your video path
        model_paths = (Path("/path/to/ov_model.xml"), Path("/path/to/ov_int8_model.xml"))  # Replace with your model paths
        zones_config_file = "path/to/zones_config.json"  # Replace with your (zones) config file path
        people_limit = 3
        model_name = "your_model_name"  # Replace with your model name
        colorful = False
        
        try:
            run(video_path, model_paths, zones_config_file, people_limit, model_name, colorful)
        except Exception as e:
            self.fail(f"End-to-end demo test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()


    