# GlitchBench - Benchmark

This repository contains the code and the data for the paper `GlitchBench: Can large multimodal models detect video game glitches?`


<div align="center">    
    <p style="font-size: 45px;"> by 
        <a href="https://taesiri.ai">Mohammad Reza Taesiri</a>, 
        Tianjun Feng
        <a href="https://anhnguyen.me/research/">Anh Nguyen</a>, and 
        <a href="https://asgaard.ece.ualberta.ca/">Cor-Paul Bezemer</a> 
    </p>
</div>


## Abstract

Large multimodal models (LMMs) have evolved from large language models (LLMs) to integrate multiple input modalities, such as visual inputs. This integration augments the capacity of LLMs in tasks requiring visual comprehension and reasoning. However, the extent and limitations of their enhanced abilities are not fully understood. To address this gap, we introduce GlitchBench, a novel benchmark designed to test and evaluate the common-sense reasoning and visual recognition capabilities of large multimodal models. Our dataset is curated from a variety of unusual, infrequent, and glitched scenarios from video game content and aims to challenge both the visual and linguistic reasoning powers of LMMs in detecting and interpreting out-of-the-ordinary events and scene composition.


## Benchmark

As the judge, we use the `Llama-2-70B` model hosted by the [Perplexity.ai](https://docs.perplexity.ai/docs/model-cards). The code for the benchmark can be found in the [./src/benchmark.py](./src/benchmark.py) file. To prevent data contamination, we will not be releasing the ground truth labels publicly.
