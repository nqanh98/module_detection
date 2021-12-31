import torch
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
from tqdm import tqdm

from solarnet.preprocessing import MaskMaker, ImageSplitter, OrthoSplitter
from solarnet.datasets import ClassifierDataset, SegmenterDataset, make_masks
from solarnet.models import Classifier, Segmenter, train_classifier, train_segmenter

from .datasets.utils import denormalize

class RunTask:

    @staticmethod
    def make_masks(data_folder='data'):
        """Saves masks for each .tif image in the raw dataset. Masks are saved
        in  <org_folder>_mask/<org_filename>.npy where <org_folder> should be the
        city name, as defined in `data/README.md`.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        """
        mask_maker = MaskMaker(data_folder=Path(data_folder))
        mask_maker.process()

    @staticmethod
    def split_images(data_folder='data', imsize=224, empty_ratio=2):
        """Generates images (and their corresponding masks) of height = width = imsize
        for input into the models.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.
        """
        splitter = ImageSplitter(data_folder=Path(data_folder))
        splitter.process(imsize=imsize, empty_ratio=empty_ratio)

    @staticmethod
    def split_ortho(ortho_folder, data_folder='data/processed', ortho_split_size=448, imsize=224, prediction=False):
        """Generates images (and their corresponding masks) of height = width = imsize
        for input into the models.
        Input Images are ORTHO image. They are cropped by ortho_split_size*ortho_split_size and resized to imsize*imsize image.

        Parameters
        ----------
        ortho_folder: str
            Path of the data folder, which should be set up such as `ortho_folder/org` and `ortho_folder/mask`
        data_folder: str
            Path of the data folder, in which cropped images are saved.
        ortho_split_size: int, default: 224
            Cropped size of ortho
        imsize: int, default: 224
            The size of the images to be generated
        prediction: bool, default: False
            If True, only org ortho data is splitted
        """
        splitter = OrthoSplitter(Path(ortho_folder), data_folder=Path(data_folder))
        splitter.process(ortho_split_size=ortho_split_size, imsize=imsize, use_prediction=prediction)

    @staticmethod
    def train_classifier(max_epochs=100, warmup=2, patience=5, val_size=0.1,
                         test_size=0.1, data_folder='data',
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the classifier

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """
        data_folder = Path(data_folder)

        model = Classifier()
        if device.type != 'cpu': model = model.cuda()

        processed_folder = data_folder / 'processed'
        dataset = ClassifierDataset(processed_folder=processed_folder)

        # make a train and val set
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(ClassifierDataset(mask=val_mask,
                                                      processed_folder=processed_folder,
                                                      transform_images=False),
                                    batch_size=64, shuffle=True)
        test_dataloader = DataLoader(ClassifierDataset(mask=test_mask,
                                                       processed_folder=processed_folder,
                                                       transform_images=False),
                                     batch_size=64)

        train_classifier(model, train_dataloader, val_dataloader, max_epochs=max_epochs,
                         warmup=warmup, patience=patience)

        savedir = data_folder / 'models'
        if not savedir.exists(): savedir.mkdir()
        torch.save(model.state_dict(), savedir / 'classifier.model')

        # save predictions for analysis
        print("Generating test results")
        preds, true = [], []
        with torch.no_grad():
            for test_x, y in tqdm(test_dataloader):
                test_preds = model(test_x)
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(y.cpu().numpy())

        np.save(savedir / 'classifier_preds.npy', np.concatenate(preds))
        np.save(savedir / 'classifier_true.npy', np.concatenate(true))

    @staticmethod
    def train_segmenter(max_epochs=100, val_size=0.1, test_size=0.1, warmup=2,
                        patience=5, model_dir='models', data_folder='data/processed/solar', use_classifier=True,
                        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the segmentation model

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        use_classifier: boolean, default: True
            Whether to use the pretrained classifier (saved in data/models/classifier.model by the
            train_classifier step) as the weights for the downsampling step of the segmentation
            model
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """
        processed_folder = Path(data_folder)
        model = Segmenter()
        if device.type != 'cpu': model = model.cuda()

        model_dir = Path(model_dir)
        if use_classifier:
            classifier_sd = torch.load(model_dir / 'classifier.model')
            model.load_base(classifier_sd)

        dataset = SegmenterDataset(processed_folder=processed_folder)
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(SegmenterDataset(mask=val_mask,
                                                    processed_folder=processed_folder,
                                                    transform_images=False),
                                    batch_size=64, shuffle=True)
        test_dataloader = DataLoader(SegmenterDataset(mask=test_mask,
                                                    processed_folder=processed_folder,
                                                    transform_images=False),
                                    batch_size=64)

        train_segmenter(model, train_dataloader, val_dataloader, max_epochs=max_epochs,
                        warmup=warmup, patience=patience)

        if not model_dir.exists(): model_dir.mkdir()
        torch.save(model.state_dict(), model_dir / 'segmenter.model')

        print("Generating test results")
        images, preds, true = [], [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                images.append(test_x.cpu().numpy())
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(model_dir / 'segmenter_images.npy', np.concatenate(images))
        np.save(model_dir / 'segmenter_preds.npy', np.concatenate(preds))
        np.save(model_dir / 'segmenter_true.npy', np.concatenate(true))

    def train_both(self, c_max_epochs=100, c_warmup=2, c_patience=5, c_val_size=0.1,
                   c_test_size=0.1, s_max_epochs=100, s_warmup=2, s_patience=5,
                   s_val_size=0.1, s_test_size=0.1, data_folder='data',
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the classifier, and use it to train the segmentation model.
        """
        data_folder = Path(data_folder)
        self.train_classifier(max_epochs=c_max_epochs, val_size=c_val_size, test_size=c_test_size,
                              warmup=c_warmup, patience=c_patience, data_folder=data_folder,
                              device=device)
        self.train_segmenter(max_epochs=s_max_epochs, val_size=s_val_size, test_size=s_test_size,
                             warmup=s_warmup, patience=s_patience, use_classifier=True,
                             data_folder=data_folder, device=device)

    @staticmethod
    def segmentation_pred(model_folder='data/models', data_folder='data/processed/solar', device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """
        segmentation prediction method
        predicts preprocessed image npy data

        Args:
            model_folder (str, optional): model folder path which have segmenter model. Defaults to 'data/model'.
            data_folder (str, optional): data folder path which have preprocessed data and under which 'solar' dir is used. Defaults to 'data'.
            device (torch.device, optional): cuda if available, else cpu
            The device to predict. Defaults to torch.device('cuda:0' if torch.cuda.is_available() else 'cpu').
        """
        model_dir = Path(model_folder)
        model = Segmenter()
        if device.type != 'cpu': model = model.cuda()

        model.load_state_dict(torch.load(model_dir / 'segmenter.model'))
        print(model.eval())

        processed_folder = Path(data_folder)

        dataloader = DataLoader(SegmenterDataset(
                                                    processed_folder=processed_folder,
                                                    transform_images=False),
                                                    batch_size=1)
        images, preds, true = [], [], []
        pred_dir = processed_folder / 'predictions'
        cnt = 0
        import cv2
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                pred = model(x)
                x = x.cpu().numpy()
                pred = pred.squeeze(1).cpu().numpy()
                y = y.cpu().numpy()
                images.append(x)
                preds.append(pred)
                true.append(y)
                _, p0 = cv2.threshold(pred[0], 0.5, 255, cv2.THRESH_BINARY)
                _, t0 = cv2.threshold(y[0], 0, 255, cv2.THRESH_BINARY)
                # print(x[0].astype(np.uint8).transpose())
                x0 = denormalize(x[0]).transpose()
                p0 = p0.transpose()
                t0 = t0.transpose()
                # print(x0.shape, p0.shape, t0.shape)
                # print(x0)
                # print(p0)
                # print(t0)
                cv2.imwrite(str(pred_dir / f"{cnt}_image.png"), x0)
                cv2.imwrite(str(pred_dir / f"{cnt}_pred.png"), p0)
                cv2.imwrite(str(pred_dir / f"{cnt}_true.png"), t0)
                cnt += 1
        
        np.save(pred_dir / 'segmenter_images.npy', np.concatenate(images))
        np.save(pred_dir / 'segmenter_preds.npy', np.concatenate(preds))
        np.save(pred_dir / 'segmenter_true.npy', np.concatenate(true))

