python prepare_big_data_for_punctuation_capitalization_task_simple.py \
  --output_dir /media/apeganov/DATA/punctuation_and_capitalization/intact/simplest/news_crawl_x2_17.01.2022 \
  --corpus_types news-crawl \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --sequence_length_range 3 128 \
  --allowed_punctuation '.,?' \
  --only_first_punctuation_character_after_word_in_autoregressive \
  --no_label_if_all_characters_are_upper_case \
  --input_files_or_dirs ~/data/wmt_raw/news-crawl/en \
  --num_jobs 24 \
  --dev_size 0 \
  --test_size 0 \
  --intact_sentences \
  --resume_from cutting