from pathlib import Path

import torch

from train import Trainer


def full_test_on_models():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 1e-5}
    model_lst = ['mse', 'gan', 'full']
    for model_type in model_lst:
        trainer = Trainer(crop_size=150, epochs=120, gen_optimizer=gen_optimizer,
                          disc_optimizer=disc_optimizer,
                          gen_optimizer_params=gen_optimizer_params,
                          disc_optimizer_params=disc_optimizer_params,
                          model_type=model_type, save_interval=20)
        trainer.fit(train_dir, eval_dir, batch_size=24,
                    data_augmentation_type='extended',
                    model_tag=model_type)


def full_test_on_augmentations():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 2e-5}
    aug_lst = ['plain', 'extended', 'photo', 'game', 'video']
    for aug_type in aug_lst:
        trainer = Trainer(crop_size=150, epochs=220, gen_optimizer=gen_optimizer,
                          disc_optimizer=disc_optimizer,
                          gen_optimizer_params=gen_optimizer_params,
                          disc_optimizer_params=disc_optimizer_params,
                          model_type='full', save_interval=25)
        trainer.fit(train_dir, eval_dir, batch_size=16,
                    data_augmentation_type=aug_type,
                    model_tag=aug_type)


def minor_test():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 2e-5}
    trainer = Trainer(crop_size=150, epochs=20, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params)
    trainer.fit(train_dir, eval_dir, batch_size=16,
                data_augmentation_type='plain')


def game_test():
    train_dir = Path('data/no_mans_sky_1080/train')
    eval_dir = Path('data/no_mans_sky_1080/valid')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.AdamW
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 2e-5}
    trainer = Trainer(crop_size=150, epochs=400, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params)
    trainer.fit(train_dir, eval_dir, batch_size=20,
                data_augmentation_type='game')


def good_attempt_run_0():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 4e-4}
    disc_optimizer_params = {'lr': 3e-5}
    g_name = None
    d_name = None
    trainer = Trainer(crop_size=100, epochs=80, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params,
                      gen_model_name=g_name,
                      disc_model_name=d_name,
                      save_interval=10)
    trainer.fit(train_dir, eval_dir, batch_size=64,
                data_augmentation_type='extended',
                model_tag='run0')


def good_attempt_run_1():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 3e-4}
    disc_optimizer_params = {'lr': 1e-4}
    loss_coeffs = (1, 0.03, 0.04)
    g_name = 'generator_run1.pt'
    d_name = 'discriminator_run1.pt'
    trainer = Trainer(crop_size=100, epochs=80, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params,
                      gen_model_name=g_name,
                      disc_model_name=d_name,
                      save_interval=10)
    trainer.fit(train_dir, eval_dir, batch_size=64,
                data_augmentation_type='extended',
                model_tag='run2')


def good_attempt_run_2():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 2e-4}
    disc_optimizer_params = {'lr': 2e-4}
    loss_coeffs = (1, 0.06, 0.1)
    g_name = 'generator_run2.pt'
    d_name = 'discriminator_run2.pt'
    trainer = Trainer(crop_size=100, epochs=80, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params,
                      gen_model_name=g_name,
                      disc_model_name=d_name,
                      save_interval=10,
                      loss_coeffs=loss_coeffs)
    trainer.fit(train_dir, eval_dir, batch_size=64,
                data_augmentation_type='extended',
                model_tag='run3')


def main():
    good_attempt_run_2()


if __name__ == "__main__":
    main()
