import os
import sys
import shutil
import urllib.request
import zipfile
import gdown

from main import song_cover_pipeline
from audio_effects import add_audio_effects
from modules.model_management import ignore_files, update_models_list, extract_zip, download_from_url, upload_zip_model, upload_separate_files
from modules.file_processing import process_file_upload

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <command> [options]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list_models":
        voice_models = ignore_files(rvc_models_dir)
        print("Available models:")
        for model in voice_models:
            print(f"- {model}")

    elif command == "convert":
        if len(sys.argv) < 6:
            print("Usage: python script.py convert <model_name> <input_audio_path> <output_format> <pitch_adjustment>")
            sys.exit(1)

        model_name = sys.argv[2]
        input_audio_path = sys.argv[3]
        output_format = sys.argv[4]
        pitch_adjustment = float(sys.argv[5])

        converted_voice = song_cover_pipeline(input_audio_path, model_name, pitch_adjustment, 
                                              index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, 
                                              f0_method='rmvpe+', crepe_hop_length=128, 
                                              protect=0.33, output_format=output_format)
        print(f"Converted voice saved to: {converted_voice}")

    elif command == "process_audio":
        if len(sys.argv) < 6:
            print("Usage: python script.py process_audio <vocal_audio_path> <instrumental_audio_path> <output_format>")
            sys.exit(1)

        vocal_audio_path = sys.argv[2]
        instrumental_audio_path = sys.argv[3]
        output_format = sys.argv[4]

        # Use default effect values or modify them as needed
        ai_cover = add_audio_effects(vocal_audio_path, instrumental_audio_path, 
                                     reverb_rm_size=0.15, reverb_wet=0.1, reverb_dry=0.8, 
                                     reverb_damping=0.7, reverb_width=1.0, 
                                     low_shelf_gain=0, high_shelf_gain=0, 
                                     compressor_ratio=4, compressor_threshold=-16, 
                                     noise_gate_threshold=-30, noise_gate_ratio=6, 
                                     noise_gate_attack=10, noise_gate_release=100, 
                                     chorus_rate_hz=0, chorus_depth=0, chorus_centre_delay_ms=0, 
                                     chorus_feedback=0, chorus_mix=0, 
                                     output_format=output_format, 
                                     vocal_gain=0, instrumental_gain=0)

        print(f"Processed AI cover saved to: {ai_cover}")

    elif command == "download_model":
        if len(sys.argv) < 4:
            print("Usage: python script.py download_model <model_download_link> <model_name>")
            sys.exit(1)

        model_download_link = sys.argv[2]
        model_name = sys.argv[3]

        download_output = download_from_url(model_download_link, model_name)
        print(download_output)

    elif command == "upload_zip_model":
        if len(sys.argv) < 4:
            print("Usage: python script.py upload_zip_model <zip_file_path> <model_name>")
            sys.exit(1)

        zip_file_path = sys.argv[2]
        model_name = sys.argv[3]

        upload_output = upload_zip_model(zip_file_path, model_name)
        print(upload_output)

    elif command == "upload_files":
        if len(sys.argv) < 5:
            print("Usage: python script.py upload_files <pth_file_path> <index_file_path> <model_name>")
            sys.exit(1)

        pth_file_path = sys.argv[2]
        index_file_path = sys.argv[3]
        model_name = sys.argv[4]

        upload_output = upload_separate_files(pth_file_path, index_file_path, model_name)
        print(upload_output)

    else:
        print("Unknown command. Available commands: list_models, convert, process_audio, download_model, upload_zip_model, upload_files")
        sys.exit(1)


if __name__ == '__main__':
    main()
