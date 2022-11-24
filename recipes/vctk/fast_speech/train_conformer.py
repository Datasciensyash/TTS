import os

from trainer import Trainer, TrainerArgs

from TTS.callbacks.weights import WeightNormCallback

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(formatter="vctk", meta_file_train="", path=os.path.join(output_path, "../VCTK/"))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = FastSpeechConfig(
    run_name="conformer_fast_speech_witcher",
    model_args=ForwardTTSArgs(
        use_pitch=False,
        encoder_type="conformer",
        encoder_params={'attention_dim': 256, 'attention_heads': 4, 'linear_units': 1536},
        decoder_type="conformer",
        decoder_params={'attention_dim': 256, 'attention_heads': 4, 'linear_units': 1536},
        d_vector_dim=512,
        hidden_channels=256,
        use_d_vector_file=True,
        d_vector_file='/home/frappuccino/dsyash/TTS/recipes/vctk/fast_speech/speaker_speechbrain_witcher.pth'
    ),
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    min_text_len=0,
    max_text_len=200,
    min_audio_len=0,
    max_audio_len=192000,
    output_path=output_path,
    datasets=[dataset_config],
#    use_speaker_embedding=True,
    lr=1.e-2,
    scheduler_after_epoch=False
)

## INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager(d_vectors_file_path='/home/frappuccino/dsyash/TTS/recipes/vctk/fast_speech/speaker_speechbrain_witcher.pth')
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers

# init model
model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

model.load_checkpoint(config, '/home/frappuccino/dsyash/TTS/recipes/vctk/fast_speech/conformer_fast_speech_witcher-November-13-2022_09+29PM-c9a7e9c8/checkpoint_910000.pth')
# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples,
    callbacks={"on_after_backward": WeightNormCallback()}
)

# AND... 3,2,1... 🚀
trainer.fit()