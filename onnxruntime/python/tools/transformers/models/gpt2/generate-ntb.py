# %%
import torch
from gpt2_beamsearch_helper import BloomLMHeadModel_BeamSearchStep, Gpt2BeamSearchHelper, Gpt2BeamSearchInputs
from gpt2_helper import Gpt2Inputs
from packaging import version
from transformers import AutoConfig

from onnxruntime import __version__ as ort_verison

model_name_or_path = "bigscience/bloom-350m"
config = AutoConfig.from_pretrained(model_name_or_path)
device = torch.device("cpu")

num_attention_heads = config.num_attention_heads
hidden_size = config.n_embed
num_layer = config.n_layer

# %%
onnx_model_path = "/home/nouamane/projects/onnxruntime/onnx_models/cpu/bigscience_bloom-350m/bloom-model.onnx"

import numpy
from transformers import AutoTokenizer

# %%
import onnxruntime

EXAMPLE_Text = ["My name is Philipp and I live in Germany."]


def get_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # okenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def get_example_inputs(prompt_text=EXAMPLE_Text):
    tokenizer = get_tokenizer(model_name_or_path)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int64)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, 0, num_attention_heads, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, empty_past


input_ids, attention_mask, empty_past = get_example_inputs()
beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
input_log_probs = torch.zeros([input_ids.shape[0], 1])
input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
prev_step_scores = torch.zeros([input_ids.shape[0], 1])

session = onnxruntime.InferenceSession(
    onnx_model_path,
    # providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    providers=["CPUExecutionProvider"],
)
ort_inputs = {
    "input_ids": numpy.ascontiguousarray(input_ids.cpu().numpy()),
    "attention_mask": numpy.ascontiguousarray(attention_mask.cpu().numpy()),
    "beam_select_idx": numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
    "input_log_probs": numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
    "input_unfinished_sents": numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
    "prev_step_results": numpy.ascontiguousarray(input_ids.cpu().numpy()),
    "prev_step_scores": numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
}
for i, past_i in enumerate(empty_past):
    ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())
ort_outputs = session.run(None, ort_inputs)
ort_outputs

# %%
# ## Pytorch Inference ##

model = BloomLMHeadModel_BeamSearchStep.from_pretrained(
    model_name_or_path,
    config=config,
    batch_size=1,
    beam_size=4,
    # cache_dir=cache_dir,
)
inputs = Gpt2BeamSearchInputs(
    input_ids,
    empty_past,
    None,
    attention_mask,
    beam_select_idx,
    input_log_probs,
    input_unfinished_sents,
    input_ids,
    prev_step_scores,
)
outputs = Gpt2BeamSearchHelper.pytorch_inference(model, inputs)

outputs

# %% [markdown]
# ## ONNX Runtime Inference with IO Binding ##
#
# To avoid data copy for input and output, ONNX Runtime also supports IO Binding. User could provide some buffer for input and outputs. For GPU inference, the buffer can be in GPU to reduce memory copy between CPU and GPU. This is helpful for high performance inference in GPU. For GPT-2, IO Binding might help the performance when batch size or (past) sequence length is large.


def inference_with_io_binding(
    session,
    config,
    input_ids,
    attention_mask,
    past,
    beam_select_idx,
    input_log_probs,
    input_unfinished_sents,
    prev_step_results,
    prev_step_scores,
    step,
    context_len,
):
    output_shapes = Gpt2BeamSearchHelper.get_output_shapes(
        batch_size=1,
        context_len=context_len,
        past_sequence_length=past[0].size(2),
        sequence_length=input_ids.size(1),
        beam_size=4,
        step=step,
        config=config,
        model_class="BloomLMHeadModel_BeamSearchStep",
    )
    output_buffers = Gpt2BeamSearchHelper.get_output_buffers(output_shapes, device)
    print("output_shapes", output_shapes)

    io_binding = Gpt2BeamSearchHelper.prepare_io_binding(
        session,
        input_ids,
        None,  # position ids
        attention_mask,
        past,
        output_buffers,
        output_shapes,
        beam_select_idx,
        input_log_probs,
        input_unfinished_sents,
        prev_step_results,
        prev_step_scores,
    )
    session.run_with_iobinding(io_binding)

    outputs = Gpt2BeamSearchHelper.get_outputs_from_io_binding_buffer(
        session, output_buffers, output_shapes, return_numpy=False
    )
    return outputs


# %% [markdown]
# We can see that the result is exactly same with/without IO Binding:

input_ids, attention_mask, empty_past = get_example_inputs()
beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
input_log_probs = torch.zeros([input_ids.shape[0], 1])
input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
prev_step_scores = torch.zeros([input_ids.shape[0], 1])
outputs = inference_with_io_binding(
    session,
    config,
    input_ids,
    attention_mask,
    empty_past,
    beam_select_idx,
    input_log_probs,
    input_unfinished_sents,
    input_ids,
    prev_step_scores,
    0,
    input_ids.shape[-1],
)
assert torch.eq(outputs[-2], torch.from_numpy(ort_outputs[-2])).all()
print("IO Binding result is good")

# %% [markdown]
# ## Batch Text Generation ##
#
# Here is an example for text generation using ONNX Runtime with/without IO Binding.


def update(output, step, batch_size, beam_size, context_length, prev_attention_mask, device):
    """
    Update the inputs for next inference.
    """
    last_state = (
        torch.from_numpy(output[0]).to(device)
        if isinstance(output[0], numpy.ndarray)
        else output[0].clone().detach().cpu()
    )

    input_ids = last_state.view(batch_size * beam_size, -1).to(device)

    input_unfinished_sents_id = -3
    prev_step_results = (
        torch.from_numpy(output[-2]).to(device)
        if isinstance(output[-2], numpy.ndarray)
        else output[-2].clone().detach().to(device)
    )

    if prev_attention_mask.shape[0] != (batch_size * beam_size):
        prev_attention_mask = prev_attention_mask.repeat(batch_size * beam_size, 1)
    attention_mask = torch.cat(
        [
            prev_attention_mask,
            torch.ones([batch_size * beam_size, 1]).type_as(prev_attention_mask),
        ],
        1,
    ).to(device)

    beam_select_idx = (
        torch.from_numpy(output[input_unfinished_sents_id - 2]).to(device)
        if isinstance(output[input_unfinished_sents_id - 2], numpy.ndarray)
        else output[input_unfinished_sents_id - 2].clone().detach().to(device)
    )
    input_log_probs = (
        torch.from_numpy(output[input_unfinished_sents_id - 1]).to(device)
        if isinstance(output[input_unfinished_sents_id - 1], numpy.ndarray)
        else output[input_unfinished_sents_id - 1].clone().detach().to(device)
    )
    input_unfinished_sents = (
        torch.from_numpy(output[input_unfinished_sents_id]).to(device)
        if isinstance(output[input_unfinished_sents_id], numpy.ndarray)
        else output[input_unfinished_sents_id].clone().detach().to(device)
    )
    prev_step_scores = (
        torch.from_numpy(output[-1]).to(device)
        if isinstance(output[-1], numpy.ndarray)
        else output[-1].clone().detach().to(device)
    )

    past = []
    if isinstance(output[1], tuple):  # past in torch output is tuple
        past = list(output[1])
    else:
        for i in range(config.n_layer):
            past_i = (
                torch.from_numpy(output[i + 1])
                if isinstance(output[i + 1], numpy.ndarray)
                else output[i + 1].clone().detach()
            )
            past.append(past_i.to(device))

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "beam_select_idx": beam_select_idx,
        "input_log_probs": input_log_probs,
        "input_unfinished_sents": input_unfinished_sents,
        "prev_step_results": prev_step_results,
        "prev_step_scores": prev_step_scores,
    }
    ort_inputs = {
        "input_ids": numpy.ascontiguousarray(input_ids.cpu().numpy()),
        "attention_mask": numpy.ascontiguousarray(attention_mask.cpu().numpy()),
        "beam_select_idx": numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
        "input_log_probs": numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
        "input_unfinished_sents": numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
        "prev_step_results": numpy.ascontiguousarray(prev_step_results.cpu().numpy()),
        "prev_step_scores": numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
    }
    for i, past_i in enumerate(past):
        ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())

    return inputs, ort_inputs, past


def test_generation(tokenizer, input_text, use_onnxruntime_io, ort_session=None, num_tokens_to_produce=30):
    print("Text generation using", "OnnxRuntime with IO binding" if use_onnxruntime_io else "OnnxRuntime", "...")
    input_ids, attention_mask, past = get_example_inputs(input_text)
    beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
    input_log_probs = torch.zeros([input_ids.shape[0], 1])
    input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
    prev_step_scores = torch.zeros([input_ids.shape[0], 1])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "beam_select_idx": beam_select_idx,
        "input_log_probs": input_log_probs,
        "input_unfinished_sents": input_unfinished_sents,
        "prev_step_results": input_ids,
        "prev_step_scores": prev_step_scores,
    }
    ort_inputs = {
        "input_ids": numpy.ascontiguousarray(input_ids.cpu().numpy()),
        "attention_mask": numpy.ascontiguousarray(attention_mask.cpu().numpy()),
        "beam_select_idx": numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
        "input_log_probs": numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
        "input_unfinished_sents": numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
        "prev_step_results": numpy.ascontiguousarray(input_ids.cpu().numpy()),
        "prev_step_scores": numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
    }
    for i, past_i in enumerate(past):
        ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())
    batch_size = input_ids.size(0)
    beam_size = 4
    context_length = input_ids.size(-1)

    for step in range(num_tokens_to_produce):
        if use_onnxruntime_io:
            outputs = inference_with_io_binding(
                ort_session,
                config,
                inputs["input_ids"],
                inputs["attention_mask"],
                past,
                inputs["beam_select_idx"],
                inputs["input_log_probs"],
                inputs["input_unfinished_sents"],
                inputs["prev_step_results"],
                inputs["prev_step_scores"],
                step,
                context_length,
            )
        else:
            outputs = ort_session.run(None, ort_inputs)
        inputs, ort_inputs, past = update(
            outputs, step, batch_size, beam_size, context_length, inputs["attention_mask"], device
        )

        if not inputs["input_unfinished_sents"].any():
            break

    print("------------")
    print(tokenizer.decode(inputs["prev_step_results"][0], skip_special_tokens=True), end="")
    print(tokenizer.decode(outputs[0][0], skip_special_tokens=True))


# %%
tokenizer = get_tokenizer(model_name_or_path)
input_text = EXAMPLE_Text
test_generation(tokenizer, input_text, use_onnxruntime_io=False, ort_session=session)

# %% [markdown]
# Next, we use ONNX Runtime with IO binding to run again and we can see that the result is exactly same.

test_generation(tokenizer, input_text, use_onnxruntime_io=True, ort_session=session)

# %%
