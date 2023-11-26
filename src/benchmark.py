import argparse
import json
import os
import pickle
import shutil
import time
from glob import glob
from multiprocessing import Manager, Process
from multiprocessing.dummy import Pool as ThreadPool
from threading import Semaphore
from urllib.parse import parse_qs, unquote, urlparse
from functools import wraps
import random
import numpy as np
import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from datasets import load_dataset


PPLX_API_KEY = os.environ["PPLX_API_KEY"]  # Perplexity API key

# Set up the rate limiter
rate_limit_tokens = 5
rate_limit_refresh_rate = 60 / rate_limit_tokens
rate_limit_semaphore = Semaphore(rate_limit_tokens)


# Decorator for rate limiting
def rate_limiter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rate_limit_semaphore.acquire()
        try:
            return func(*args, **kwargs)
        finally:
            time.sleep(rate_limit_refresh_rate)  # Wait for the rate limit to refresh
            rate_limit_semaphore.release()

    return wrapper


@rate_limiter
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_query(gt_text, lmm_output, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Your task is to compare a model-generated text with a ground truth reference, assessing whether the key information and themes are similarly conveyed, even if worded differently. Focus on semantic content, thematic alignment, and intent, rather than exact phrasing or word usage. Recognize synonyms, paraphrases, and different stylistic expressions as valid, provided they faithfully represent the ground truth's meaning. Offer feedback on the correlation between the texts and suggest improvements for alignment, while appreciating creative or varied linguistic expression that maintains the essence of the ground truth.\n\nFirst analyze, then report the final answer in either of Yes or No",
            },
            {
                "role": "user",
                "content": f"Ground truth:\n\n{gt_text}\n\nGenerated Description: \n\n{lmm_output}",
            },
        ],
        api_base="https://api.perplexity.ai",
        api_key=PPLX_API_KEY,
    )
    answer = response["choices"][0]["message"]["content"]
    return answer, response


def writer(output_file, queue):
    # Initialize dictionaries or load existing data
    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                data = pickle.load(f)
                query_inputs = data.get("query_inputs", {})
                llm_responses = data.get("llm_responses", {})
                llm_full_responses = data.get("llm_full_responses", {})
        except Exception as e:
            print(f"Failed to load existing data from {output_file}: {e}")
            query_inputs = {}
            llm_responses = {}
            llm_full_responses = {}
    else:
        query_inputs = {}
        llm_responses = {}
        llm_full_responses = {}

    while True:
        item = queue.get()
        if item is None:
            print("Writer received sentinel value, stopping.")  # Debugging line
            break

        local_query_inputs, local_llm_responses, local_llm_full_responses = item

        # Update the dictionaries with new data
        query_inputs.update(local_query_inputs)
        llm_responses.update(local_llm_responses)
        llm_full_responses.update(local_llm_full_responses)

        # Save the updated data back to the file
        with open(output_file, "wb") as f:
            combined_data = {
                "query_inputs": query_inputs,
                "llm_responses": llm_responses,
                "llm_full_responses": llm_full_responses,
            }
            pickle.dump(combined_data, f)

    print(f"Writer has completed writing to {output_file}")


def process_sample(image, prompt, model, queue, pbar):
    unique_key = f"{image}_{prompt}"  # Create a unique key using both image and prompt

    local_query_inputs = {}
    local_llm_responses = {}
    local_llm_full_responses = {}

    try:
        image_gt = gt_texts[image]
        llm_output = lmm_output_df[
            (lmm_output_df["id"] == image) & (lmm_output_df["prompt"] == prompt)
        ]["Output"].values[0]

        # remove trailing <s> and </s> tokens
        prompt_content = llm_output.replace("<s>", "").replace("</s>", "")

        # send data to judge
        x, y = chat_query(image_gt, llm_output, model=model)

        local_query_inputs[unique_key] = prompt_content
        local_llm_responses[unique_key] = x
        local_llm_full_responses[unique_key] = y

        # Pass a tuple of dictionaries to the queue
        queue.put((local_query_inputs, local_llm_responses, local_llm_full_responses))
    except Exception as e:
        print(f"Error for image: {image} with prompt: {prompt}")  # Debugging line
        print(e)
        raise
    finally:
        pbar.update(1)  # Update the progress bar


def main():
    lmm_name = "llava-1.5-7b"
    lmm_output_df = pd.read_csv("./lmm-output.csv")

    glitchbench_dataset = load_dataset("glitchbench/GlitchBenchPrivate")["validation"]
    gt_texts = {x["id"]: x["description"] for x in glitchbench_dataset}

    all_prompts = lmm_output_df["prompt"].unique().tolist()
    all_images = lmm_output_df["id"].unique().tolist()

    judge_model = "llama-2-70b-chat"
    output_file = f"./{judge_model}-judges-{lmm_name}.pkl"

    query_inputs = {}
    judge_responses = {}
    judge_full_responses = {}

    # Load previous data if the output file exists
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            previous_data = pickle.load(f)
            query_inputs = previous_data.get("query_inputs", {})
            judge_responses = previous_data.get("judge_responses", {})
            judge_full_responses = previous_data.get("judge_full_responses", {})

    print(f'Loaded {len(query_inputs)} samples from "{output_file}"')

    # Create a bag of samples (image, prompt) tuples
    sample_bag = [(image, prompt) for prompt in all_prompts for image in all_images]

    # Filter out samples that have already been processed
    processed_keys = set(query_inputs.keys())
    print(f"Found {len(processed_keys)} samples in the output file")
    sample_bag = [
        sample
        for sample in sample_bag
        if f"{sample[0]}_{sample[1]}" not in processed_keys
    ]

    print("-----------------------------")
    print("Starting to process samples")

    with Manager() as manager:
        queue = manager.Queue()  # Create a shared queue
        pbar = tqdm(total=len(sample_bag), dynamic_ncols=True)

        writer_process = Process(target=writer, args=(output_file, queue))
        writer_process.start()

        async_results = []  # Store the AsyncResult objects

        # Use more threads if needed, but for debugging, we'll use just one
        with ThreadPool(1) as pool:
            for sample in sample_bag:
                image, prompt = sample
                async_result = pool.apply_async(
                    process_sample, args=(image, prompt, judge_model, queue, pbar)
                )
                async_results.append(async_result)  # Store the AsyncResult object

            # Wait for all tasks to complete
            for async_result in async_results:
                async_result.wait()

        queue.put(None)  # Indicate that all data has been processed
        writer_process.join()  # Wait for the writer process to finish

    print("Done!")
    print("-----------------------------")
    pbar.close()


if __name__ == "__main__":
    main()
