from absl import app, flags, logging
import torch as th
import pytorch_lightning as pl
import nlp
import transformers
import sh


FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')



sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


class IMDB_sentiment(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def prepare_data(self):
        train_ds = nlp.load_dataset('imdb', split='train[:5%]')
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)

        #import IPython ; IPython.embed(); exit(1)

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )




def main(_):
    logging.info('hello')

    model = IMDB_sentiment()
    
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus= (1 if th.cuda.is_available() else 0),
        max_epochs= FLAGS.epochs
        )

    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)