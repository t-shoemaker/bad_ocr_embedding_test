About
=====

This experiment tests whether a large language model can "read past" OCR
errors. Specifically, we identify a series of sentences from a corpus that
contain OCR errors, which we then correct. Then, we send the erroneous sentence
and the fixed sentence to a model and retrieve embeddings for both, with the
hope that the these embeddings will be interchangeable.

+ Test corpus: a collection of 18-19C medical texts compiled by [Sarah
  Bull][sarah]
+ Model used: [Llama-2 7B][llm], quantized down to 4-bits with [llama.cpp][lccp]

[sarah]: https://www.torontomu.ca/english/about-us/faculty-and-staff/faculty/bull-sarah/
[llm]: https://huggingface.co/meta-llama/Llama-2-70b
[lccp]: https://github.com/ggerganov/llama.cpp

Process:

1. Run the following:
   ```
   ./src/sample_sents.py --indir /path/to/data --outfile /path/to/sample.txt
   ```
   This will yank sentences from the documents using `nltk`'s `sent_tokenize()`
   function. That's by no means a perfect method, but we just need some
   examples, so it'll do

2. Manually read the sentences and identify ones with bad OCR. Place those in a
   new plaintext file. Each sentence should be on its own line

3. Create a new plaintext file and, on each line, provide a corrected version
   of the sentences above. **Ensure that the lines of each file contain
   properly aligned sentences.**

4. Embed the sentences with the model:
   ```
   ./src/llama_embed.py \
     --uncorrected /path/to/uncorrected.txt \
     --corrected /path/to/corrected.txt \
     --embeddings /path/to/embeddings.jsonl \
     --model_path /path/to/model
   ```

5. Compare the embeddings in `test_suite.ipynb`
