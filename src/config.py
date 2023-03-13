

class Config(object):
    def __init__(self):
        # dataset
        self.dataset_path = "amazon_reviews_multi"
        self.dataset_config = "en"
        self.text_column = "review_body"
        self.label_column = "stars"
        self.num_labels = 5
        # model and tokenizer
        self.checkpoint = "distilbert-base-uncased"
        # training configs
        self.batch_size = 8
        self.epochs = 1
        self.learning_rate = 2e-5
        # wandb configs
        self.WANDB_API_KEY = ""
        self.WANDB_PROJECT = "sentiment-analysis"
        self.WANDB_ENTITY = ""

config = Config()
