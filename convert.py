import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple, List, Dict, Any

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def convert_to_onnx(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str = "model.onnx"
) -> None:
    model.eval()
    
    dummy_input: torch.Tensor = torch.randint(0, tokenizer.vocab_size, (1, 128))
    input_names: List[str] = ["input_ids"]
    output_names: List[str] = ["logits"]
    dynamic_axes: Dict[str, Dict[int, str]] = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

def validate_onnx_model(model_path: str) -> ort.InferenceSession:
    onnx_model: onnx.ModelProto = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session: ort.InferenceSession = ort.InferenceSession(model_path)
    return ort_session

def generate_text_onnx(
    ort_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 50
) -> str:
    inputs: Dict[str, torch.Tensor] = tokenizer(prompt, return_tensors="pt")
    input_ids: np.ndarray = inputs['input_ids'].numpy()
    
    generated: np.ndarray = input_ids
    for _ in range(max_length):
        outputs: List[np.ndarray] = ort_session.run(None, {'input_ids': generated})
        next_token_logits: np.ndarray = outputs[0][:, -1, :]
        next_token: torch.Tensor = torch.argmax(
            torch.from_numpy(next_token_logits), 
            dim=-1
        ).unsqueeze(0)
        generated = np.concatenate([generated, next_token.numpy()], axis=1)
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main() -> None:
    model_name: str = "Salesforce/codegen-350M-mono"
    onnx_path: str = "model.onnx"
    test_prompt: str = "def hello_world():"
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    convert_to_onnx(model, tokenizer, onnx_path)
    ort_session = validate_onnx_model(onnx_path)
    result: str = generate_text_onnx(ort_session, tokenizer, test_prompt)
    print(result)

if __name__ == "__main__":
    main()