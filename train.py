from peakdetect.data_module import DPDataModule
from peakdetect.lightning_module import EDPeakDector
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

args = dict(model='peakdetect/peakdetect_Resnet.cfg',
            data = 'peakdetect/data.cfg',
            checkpoint_path='checkpoints/orth_5_degree',
            model_name='orth_5_degree_Resnet',
            version='test',
            test_data_ratio=0.05,
            batch_size=16,
            precision_for_training=16,
            num_epoch=30,
            verbose='store_true',
            n_cpu=4,
            iou_thres=0.5,
            conf_thres=0.1,
            nms_thres=0.5,
            seed=0)

# Set Logger
logger = TensorBoardLogger(save_dir="lightning_logs",
                           name=args['model_name'],
                           version=args['version'])

# Set checkpoints paths
checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="precision",
    every_n_epochs=5,
    mode="max",
    dirpath=args['checkpoint_path'],
    filename= args['model_name'] + "-{epoch:02d}-{precission:.2f}",
)

def main():
    model = EDPeakDector(args)
    data_module = DPDataModule(args, simulation=False)
    trainer = pl.Trainer(max_epochs=args['num_epoch'], 
                         callbacks=[TQDMProgressBar(refresh_rate=20),checkpoint_callback],
                         logger=logger,
                         precision=args['precision_for_training'],
                         accelerator='gpu', 
                         devices=4,
                         strategy=None)

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()