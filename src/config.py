

class Config(object):
    def __init__(self):
        # dataset
        self.dataset_path = "amazon_reviews_multi"
        self.dataset_config = "en"
        self.text_column = "review_body"
        self.label_column = "stars"
        # model and tokenizer
        self.checkpoint = "distilbert-base-uncased"
        # training configs
        self.batch_size = 8
        self.epochs = 1
        self.learning_rate = 5e-5
        # wandb configs
        self.WANDB_KEY = ""
        self.WANDB_PROJECT = "sentiment-analysis"


config = Config()
