import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += (
                "<|assistant|>\n"
                + message["content"].strip()
                + tokenizer.eos_token
                + "\n"
            )
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    completion: str,
    max_length: int,
    print_ex: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[: len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def get_bbh_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format: bool = True,
    chat_format: str = "tulu",
    **kwargs,
):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"

    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]

        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        task_prompt = "\n\n".join(stuff[:-3])

        def form_icl(exs):
            string = ""
            for ex in exs:
                question, answer = ex.split("\nA:")
                string += question + "\nA:" + answer
                string += "\n\n"
            return string

        for i in range(len(exes)):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i + 1 :]
            icl = form_icl(other_exes)
            question, answer = target_ex.split("\nA:")

            if use_chat_format:
                if (
                    chat_format == "tulu"
                ):  # we follow the tulu instruction tuning format
                    question = (
                        "<|user|>\n"
                        + task_prompt.strip()
                        + "\n\n"
                        + icl
                        + f"{question}"
                        + "\n<|assistant|>\nA:"
                    )
                else:
                    question = (
                        f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    )
            else:
                question = task_prompt.strip() + "\n\n" + f"{question}" + "\nA:"
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer,
                question,
                answer,
                max_length,
                print_ex=True if i == 0 else False,
            )
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tydiqa_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format: bool = True,
    chat_format: str = "tulu",
    zh: bool = False,
    **kwargs,
) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": (
            "Answer the following question based on the information in the given passage.",
            "Passage:",
            "Question:",
            "Answer:",
        ),
        "arabic": (
            "أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.",
            "المقطع:",
            "السؤال:",
            "الإجابة:",
        ),
        "bengali": (
            "প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।",
            "অধ্যায়:",
            "প্রশ্ন:",
            "উত্তর:",
        ),
        "finnish": (
            "Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.",
            "Kappale:",
            "Kysymys:",
            "Vastaus:",
        ),
        "indonesian": (
            "Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.",
            "Bagian:",
            "Pertanyaan:",
            "Jawaban:",
        ),
        "korean": (
            "주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.",
            "문단:",
            "질문:",
            "답변:",
        ),
        "russian": (
            "Ответьте на следующий вопрос на основе информации в данном отрывке.",
            "Отрывок:",
            "Вопрос:",
            "Ответ:",
        ),
        "swahili": (
            "Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.",
            "Kifungu:",
            "Swali:",
            "Jibu:",
        ),
        "telugu": (
            "ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.",
            "పేరా:",
            "ప్రశ్న:",
            "సమాధానం:",
        ),
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。",
                "文章:",
                "问题:",
                "答案:",
            )

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/eval/tydiqa/dev", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[
            lang
        ]
        prompt += (
            p_template
            + " "
            + format(example["context"])
            + "\n"
            + q_template
            + " "
            + format(example["question"])
            + "\n"
        )
        answer = " " + format(example["answers"][0]["text"])
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_mmlu_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format=True,
    chat_format="tulu",
    **kwargs,
):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.


    <|user|>
    The following are multiple choice questions (with answers) about  world religions.

    What is the sign of the covenant for Jewish males?
    A. The rainbow
    B. Circumcision
    C. A son
    D. Bar mitzvah
    Answer:
    <|assistant|>
    The answer is: B
    """
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, i=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        prompt += format_example(train_df, i, include_answer=False)
        return prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        return prompt

    k = 5
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
        )[:k]
        for i in range(k):
            prompt = gen_prompt(dev_df, subject, i)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]
            print(prompt)
            print(answer)

            if use_chat_format:
                if chat_format == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                else:
                    # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
            else:
                prompt = prompt
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer,
                prompt,
                answer,
                max_length,
                print_ex=True if i == 0 else False,
            )
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_commonsenseqa_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format=True,
    chat_format="tulu",
    **kwargs,
):
    """{"question": "Bill is stuck in marsh when a man comes up to him peaking Cajun, where is he?",
    "question_concept": "marsh",
    "choices": {"label": ["A", "B", "C", "D", "E"],
    "text": ["low lands", "new york", "forest", "louisiana", "everglades"]},
    "answerKey": "D"}"""
    commonsenseqa_data_dir = os.path.join(
        data_dir, "eval", "commonsense_qa/commonsenseqa_val.jsonl"
    )

    def gen_prompt(train_df):
        prompt = format_example(train_df, include_answer=False)
        return prompt

    def format_example(data, include_answer=True):
        prompt = "Question: {}\n".format(data["question"])
        choices = data["choices"]["label"]
        texts = data["choices"]["text"]

        # 添加选项
        for i, choice in enumerate(choices):
            prompt += "{}. {}\n".format(choice, texts[i])

        # 添加答案提示
        prompt += "Answer:"

        if include_answer:
            prompt += " " + data["answerKey"]

        return prompt

    k = 5
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    alldata = []
    with open(commonsenseqa_data_dir, "r") as file:
        for f in file:
            alldata.append(json.loads(f))
    for i in range(k):
        prompt = gen_prompt(alldata[i])
        answer = " " + alldata[i]["answerKey"]

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True if i == 0 else False
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_gsm8k_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format: bool = True,
    chat_format: str = "tulu",
    zh: bool = False,
    **kwargs,
) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    file_name = "validation.jsonl"
    file = os.path.join(f"{data_dir}/eval/gsm/", file_name)

    examples = []
    with open(file, "r") as files:
        for f in files:
            examples.append(json.loads(f))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for example in examples:
        max_seq_length = 2048
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("messages field is empty.")

        example_text = concat_messages(messages, tokenizer)
        print(example_text)
        tokenized_example = tokenizer(
            example_text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
        )
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        concat_messages(messages[:message_idx], tokenizer),
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]
                if (
                    message_idx < len(messages) - 1
                    and messages[message_idx + 1]["role"] == "assistant"
                ):
                    # here we also ignore the role of the assistant
                    messages_so_far = (
                        concat_messages(messages[: message_idx + 1], tokenizer)
                        + "<|assistant|>\n"
                    )
                else:
                    messages_so_far = concat_messages(
                        messages[: message_idx + 1], tokenizer
                    )
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        dataset["input_ids"].append(input_ids.flatten())
        dataset["labels"].append(labels.flatten())
        dataset["attention_mask"].append(attention_mask.flatten())
    dataset = Dataset.from_dict(dataset)
    print(dataset)
    return dataset


def get_arcchallenge_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format=True,
    chat_format="tulu",
    **kwargs,
):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, i=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        prompt += format_example(train_df, i, include_answer=False)
        return prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        return prompt

    k = 5
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
        )[:k]
        for i in range(k):
            prompt = gen_prompt(dev_df, subject, i)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]

            if use_chat_format:
                if chat_format == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                else:
                    # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
            else:
                prompt = prompt
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer,
                prompt,
                answer,
                max_length,
                print_ex=True if i == 0 else False,
            )
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_boolq_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format: bool = True,
    chat_format: str = "tulu",
    zh: bool = False,
    **kwargs,
) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": (
            "Answer the following question based on the information in the given passage.",
            "Passage:",
            "Question:",
            "Answer:",
        ),
        "arabic": (
            "أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.",
            "المقطع:",
            "السؤال:",
            "الإجابة:",
        ),
        "bengali": (
            "প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।",
            "অধ্যায়:",
            "প্রশ্ন:",
            "উত্তর:",
        ),
        "finnish": (
            "Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.",
            "Kappale:",
            "Kysymys:",
            "Vastaus:",
        ),
        "indonesian": (
            "Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.",
            "Bagian:",
            "Pertanyaan:",
            "Jawaban:",
        ),
        "korean": (
            "주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.",
            "문단:",
            "질문:",
            "답변:",
        ),
        "russian": (
            "Ответьте на следующий вопрос на основе информации в данном отрывке.",
            "Отрывок:",
            "Вопрос:",
            "Ответ:",
        ),
        "swahili": (
            "Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.",
            "Kifungu:",
            "Swali:",
            "Jibu:",
        ),
        "telugu": (
            "ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.",
            "పేరా:",
            "ప్రశ్న:",
            "సమాధానం:",
        ),
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。",
                "文章:",
                "问题:",
                "答案:",
            )

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/eval/tydiqa/dev", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[
            lang
        ]
        prompt += (
            p_template
            + " "
            + format(example["context"])
            + "\n"
            + q_template
            + " "
            + format(example["question"])
            + "\n"
        )
        answer = " " + format(example["answers"][0]["text"])
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_hellaswag_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_format: bool = True,
    chat_format: str = "tulu",
    zh: bool = False,
    **kwargs,
) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": (
            "Answer the following question based on the information in the given passage.",
            "Passage:",
            "Question:",
            "Answer:",
        ),
        "arabic": (
            "أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.",
            "المقطع:",
            "السؤال:",
            "الإجابة:",
        ),
        "bengali": (
            "প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।",
            "অধ্যায়:",
            "প্রশ্ন:",
            "উত্তর:",
        ),
        "finnish": (
            "Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.",
            "Kappale:",
            "Kysymys:",
            "Vastaus:",
        ),
        "indonesian": (
            "Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.",
            "Bagian:",
            "Pertanyaan:",
            "Jawaban:",
        ),
        "korean": (
            "주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.",
            "문단:",
            "질문:",
            "답변:",
        ),
        "russian": (
            "Ответьте на следующий вопрос на основе информации в данном отрывке.",
            "Отрывок:",
            "Вопрос:",
            "Ответ:",
        ),
        "swahili": (
            "Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.",
            "Kifungu:",
            "Swali:",
            "Jibu:",
        ),
        "telugu": (
            "ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.",
            "పేరా:",
            "ప్రశ్న:",
            "సమాధానం:",
        ),
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。",
                "文章:",
                "问题:",
                "答案:",
            )

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/eval/tydiqa/dev", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[
            lang
        ]
        prompt += (
            p_template
            + " "
            + format(example["context"])
            + "\n"
            + q_template
            + " "
            + format(example["question"])
            + "\n"
        )
        answer = " " + format(example["answers"][0]["text"])
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_alpaca_dataset(
    data_file: str,  # *.json / *.jsonl 路径（alpaca 或 alpaca_gpt4）
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
    add_eos_token: bool = True,
    **kwargs,
):
    """
    读取 Alpaca / Alpaca-GPT-4 数据集，返回 HuggingFace `Dataset`，字段如下
    ┌──────────────┬──────────────────────────────────────────────────────────┐
    │ input_ids    │ prompt+response 整序列                                     │
    │ attention_mask│ 与 input_ids 等长                                           │
    │ labels       │ SFT 标签；prompt 部分已置 -100                                │
    │ y_mask       │ 0/1 向量；1 → 属于 y                                         │
    │ y_input_ids  │ 仅回答 token 序列                                           │
    │ y_attention_mask│ 与 y_input_ids 等长                                      │
    │ prompt_len   │ int，prompt token 数                                       │
    └──────────────┴──────────────────────────────────────────────────────────┘
    这样：
      • 计算 𝓛_PT 时直接把 `y_input_ids` 喂给模型即可
      • 计算 𝓛_IFL 时用 `labels` 得到对数似然，再减去单独算出的 𝓛_PT
    """
    # Alpaca 的模板
    # PROMPT_DICT = {
    #     "prompt_input": (
    #         "Below is an instruction that describes a task, paired with an input that provides further context. "
    #         "Write a response that appropriately completes the request.\n\n"
    #         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    #     ),
    #     "prompt_no_input": (
    #         "Below is an instruction that describes a task. "
    #         "Write a response that appropriately completes the request.\n\n"
    #         "### Instruction:\n{instruction}\n\n### Response:"
    #     ),
    # }

    PROMPT_DICT = {
        "prompt_input": ("<|user|>\n{instruction}\n{input}\n\n<|assistant|>\n"),
        "prompt_no_input": ("<|user|>\n{instruction}\n\n<|assistant|>\n"),
    }

    # 读取 json / jsonl
    raw_examples = []
    with open(data_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.strip().startswith("{"):
            # jsonl
            raw_examples = [json.loads(line) for line in f]
        else:
            # 普通 json
            raw_examples = json.load(f)

    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "y_mask": [],
        "y_input_ids": [],
        "y_attention_mask": [],
        "prompt_len": [],
    }

    for example in raw_examples:
        instruction = example["instruction"].strip()
        inp = example.get("input", "").strip()
        output = example["output"].strip()

        # -------- 1) 组 prompt --------
        if inp:
            prompt = PROMPT_DICT["prompt_input"].format(
                instruction=instruction, input=inp
            )
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

        completion = " " + output  # 模型回答前面加空格符合 Alpaca 原实现

        # -------- 2) Tokenize --------
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True)
        full_prompt = prompt + completion
        if (
            add_eos_token
            and tokenizer.eos_token
            and not full_prompt.endswith(tokenizer.eos_token)
        ):
            full_prompt += tokenizer.eos_token
            completion += tokenizer.eos_token

        full_ids = tokenizer.encode(full_prompt, max_length=max_length, truncation=True)
        labels = full_ids.copy()
        # labels[: len(prompt_ids)] = [-100] * len(prompt_ids)  # 屏蔽 prompt 区域
        labels[: len(prompt_ids) + 1] = [-100] * (len(prompt_ids) + 1)
        attention_mask = [1] * len(full_ids)

        # y 区域信息
        y_ids = tokenizer.encode(
            completion, add_special_tokens=False, max_length=max_length, truncation=True
        )
        y_att = [1] * len(y_ids)
        y_mask = [0] * len(full_ids)
        start = len(prompt_ids)
        y_mask[start : start + len(y_ids)] = [1] * len(y_ids)

        # -------- 3) 填写字段 --------
        dataset_dict["input_ids"].append(torch.tensor(full_ids))
        dataset_dict["attention_mask"].append(torch.tensor(attention_mask))
        dataset_dict["labels"].append(torch.tensor(labels))
        dataset_dict["y_mask"].append(torch.tensor(y_mask))
        dataset_dict["y_input_ids"].append(torch.tensor(y_ids))
        dataset_dict["y_attention_mask"].append(torch.tensor(y_att))
        dataset_dict["prompt_len"].append(len(prompt_ids))

    return Dataset.from_dict(dataset_dict)


def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "bbh":
        return get_bbh_dataset(**kwargs)
    elif task == "tydiqa":
        return get_tydiqa_dataset(**kwargs)
    elif task == "mmlu":
        return get_mmlu_dataset(**kwargs)
    elif task == "gsm8k":
        return get_gsm8k_dataset(**kwargs)
    elif task == "boolq":
        return get_boolq_dataset(**kwargs)
    elif task == "commonsense_qa":
        return get_commonsenseqa_dataset(**kwargs)
    elif task == "arc-challenge":
        return get_arcchallenge_dataset(**kwargs)
    elif task.find("alpaca") != -1:
        print("Loading Alpaca dataset")
        return get_alpaca_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # When getting gradients, we only do this single batch process
        collate_fn=data_collator,
        shuffle=False,
    )
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
