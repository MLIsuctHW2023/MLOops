import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import os

from app import app, process_code_with_triton

class TestCodegenAPI:
    
    def setup_method(self) -> None:
        self.client: TestClient = TestClient(app)
        self.load_test_prompts()
    
    def load_test_prompts(self) -> None:
        self.prompts: dict = {}
        for i in range(1, 11):
            filename: str = f"{i}.txt"
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    self.prompts[str(i)] = f.read().strip()
    
    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_codegen_code_1(self) -> None:
        if '1' not in self.prompts:
            pytest.skip("File 1.txt not found")
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "def example_function(): pass"
            response = self.client.post("/codegen", json={"prompt": self.prompts['1']})
            assert response.status_code == 200
    
    def test_codegen_code_2(self) -> None:
        if '2' not in self.prompts:
            pytest.skip("File 2.txt not found")
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "class ExampleClass: pass"
            response = self.client.post("/codegen", json={"prompt": self.prompts['2']})
            assert response.status_code == 200
    
    def test_codegen_code_3(self) -> None:
        if '3' not in self.prompts:
            pytest.skip("File 3.txt not found")
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "for i in range(10): print(i) - tests.py:49"
            response = self.client.post("/codegen", json={"prompt": self.prompts['3']})
            assert response.status_code == 200
    
    def test_codegen_code_4(self) -> None:
        if '4' not in self.prompts:
            pytest.skip("File 4.txt not found")
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "import numpy as np"
            response = self.client.post("/codegen", json={"prompt": self.prompts['4']})
            assert response.status_code == 200
    
    def test_codegen_code_5(self) -> None:
        if '5' not in self.prompts:
            pytest.skip("File 5.txt not found")
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "if __name__ == '__main__': main()"
            response = self.client.post("/codegen", json={"prompt": self.prompts['5']})
            assert response.status_code == 200
    
    def test_codegen_empty_prompt(self) -> None:
        response = self.client.post("/codegen", json={"prompt": ""})
        assert response.status_code == 400
    
    def test_codegen_whitespace_prompt(self) -> None:
        response = self.client.post("/codegen", json={"prompt": "   "})
        assert response.status_code == 400
    
    def test_codegen_long_prompt(self) -> None:
        long_prompt: str = "def " + "x" * 2000
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "codegen_code"
            response = self.client.post("/codegen", json={"prompt": long_prompt})
            assert response.status_code == 200
    
    def test_codegen_special_chars(self) -> None:
        special_prompt: str = "def function_!@#$%^&*() -> None:"
        with patch('app.process_code_with_triton') as mock_triton:
            mock_triton.return_value = "def processed_function(): pass"
            response = self.client.post("/codegen", json={"prompt": special_prompt})
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])