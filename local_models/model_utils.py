from typing import Optional, Union, List
from ..utils.misc import all_exists
import torch
from ..utils.import_utils import is_transformers_available, is_peft_available


def TF_model_from_pretrained(
        base_model: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attn: bool = False,
        device_map='auto',
):
    if not is_transformers_available():
        raise ImportError('transformers lib not installed')

    from transformers import AutoModel, BitsAndBytesConfig

    assert not (load_in_4bit and load_in_8bit), 'both 4bit and 8bit are True, give me just one flag'

    add_kwargs = {}
    if use_flash_attn:
        add_kwargs['attn_implementation'] = "flash_attention_2"
    if load_in_4bit or load_in_8bit:
        add_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

    model = AutoModel.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        **add_kwargs
    )

    return model


def peft_model_from_pretrained(
        base_model,
        # lora params, if you init one for training
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: Optional[float] = None,
        # lora path, if you load one for inference; or if a list of (path, adapter_name) given, i will load all of them
        # note the last lora is active by default
        peft_path: Union[str, List, None] = None,
        to_device: Optional[str] = None,
):
    """

    :param base_model:
    :param lora_r: lora params, if you init one for training
    :param lora_alpha: lora params, if you init one for training
    :param lora_target_modules: lora params, if you init one for training
    :param lora_dropout: lora params, if you init one for training
    :param peft_path: lora path, if you load one for inference; or if a list of (path, adapter_name) given, i will load
        all of them, note the last lora is active by default
    :param to_device: the device to put the model, by default (i.e., None), peft loads to cpu, and you can load it to
        cuda with 'cuda'
    :return:
    """
    if not is_peft_available():
        raise ImportError('peft lib not installed')

    from peft import (
        prepare_model_for_kbit_training,
        LoraConfig,
        get_peft_model,
        PeftModel
    )

    is_init_peft = all_exists(lora_r, lora_alpha, lora_target_modules, lora_dropout)
    is_load_peft = all_exists(peft_path)
    assert not (is_init_peft and is_load_peft), (
        'you give too many args, either give peft_path for me to load one, or the rest for me to init one'
    )
    assert is_init_peft or is_load_peft, (
        'you give too few args, either give peft_path for me to load one, or the rest for me to init one'
    )

    if is_init_peft:
        model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        if all_exists(to_device):
            model.to(to_device)
        model.print_trainable_parameters()

    elif is_load_peft:
        if isinstance(peft_path, str):
            model = PeftModel.from_pretrained(
                base_model,
                peft_path,
                torch_dtype=torch.float16
            )
            if all_exists(to_device):
                model.to(to_device)

        elif isinstance(peft_path, list):
            model = base_model
            for ppath, name in peft_path:
                model = PeftModel.from_pretrained(
                    model,
                    ppath,
                    torch_dtype=torch.float16,
                    adapter_name=name
                )
            if all_exists(to_device):
                model.to(to_device)

    return model


def load_TF_model_and_peft(
        base_model,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attn: bool = False,
        device_map='auto',
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: Optional[float] = None,
        peft_path: Union[str, List, None] = None,
        to_device: Optional[str] = None,
):
    base_model = TF_model_from_pretrained(
        base_model,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_flash_attn=use_flash_attn,
        device_map=device_map,
    )

    model = peft_model_from_pretrained(
        base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        peft_path=peft_path,
        to_device=to_device,
    )

    return model