import os, sys
import shutil
import urllib.request
import zipfile
import gdown
import gradio as gr

from main import song_cover_pipeline
from audio_effects import add_audio_effects
from modules.model_management import ignore_files, update_models_list, extract_zip, download_from_url, upload_zip_model, upload_separate_files
from modules.ui_updates import show_hop_slider, update_f0_method, update_button_text, update_button_text_voc, update_button_text_inst, swap_visibility, swap_buttons
from modules.file_processing import process_file_upload

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


warning = sys.argv[1]



if __name__ == '__main__':
    voice_models = ignore_files(rvc_models_dir)

    with gr.Blocks(title='CoverGen Lite - Politrees', theme="Hev832/Applio") as app:
        gr.HTML("<center><h1>Welcome to CoverGen Lite - Politrees</h1></center>")

    
        with gr.Tab("Voice Conversion"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, variant='panel'):
                    with gr.Group():
                        rvc_model = gr.Dropdown(voice_models, label='Voice Models')
                        ref_btn = gr.Button('Refresh Models List', variant='primary')
                    with gr.Group():
                        pitch = gr.Slider(-24, 24, value=0, step=0.5, label='Pitch Adjustment', info='-24 - male voice || 24 - female voice')
                with gr.Column(scale=2, variant='panel'):
                    local_file = gr.Textbox(label='url File', interactive=False)
                    gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                            
                          
            output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='File Format', scale=0.1, allow_custom_value=False, filterable=False)
            generate_btn = gr.Button("Generate", variant='primary', scale=1)
            converted_voice = gr.Audio(label='Converted Voice', scale=5, show_share_button=False)
                    
            with gr.Accordion('Voice Conversion Settings', open=False):
                with gr.Group():
                    with gr.Column(variant='panel'):
                        use_hybrid_methods = gr.Checkbox(label="Use Hybrid Methods", value=False)
                        f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe', 'crepe'], value='rmvpe+', label='F0 Method', allow_custom_value=False, filterable=False)
                        use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                        crepe_hop_length = gr.Slider(8, 512, value=128, step=8, visible=False, label='Crepe Hop Length')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                        with gr.Row():
                            f0_min = gr.Slider(label="Minimum pitch range", info="Defines the lower limit of the pitch range that the algorithm will use to determine the fundamental frequency (F0) in the audio signal.", step=1, minimum=1, value=50, maximum=120)
                            f0_max = gr.Slider(label="Maximum pitch range", info="Defines the upper limit of the pitch range that the algorithm will use to determine the fundamental frequency (F0) in the audio signal.", step=1, minimum=380, value=1100, maximum=16000)
                    with gr.Column(variant='panel'):
                        index_rate = gr.Slider(0, 1, value=0, label='Index Rate', info='Controls the extent to which the index file influences the analysis results. A higher value increases the influence of the index file, but may amplify breathing artifacts in the audio. Choosing a lower value may help reduce artifacts.')
                        filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius', info='Manages the radius of filtering the pitch analysis results. If the filtering value is three or higher, median filtering is applied to reduce breathing noise in the audio recording.')
                        rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.01, label='RMS Mix Rate', info='Controls the extent to which the output signal is mixed with its envelope. A value close to 1 increases the use of the envelope of the output signal, which may improve sound quality.')
                        protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label='Consonant Protection', info='Controls the extent to which individual consonants and breathing sounds are protected from electroacoustic breaks and other artifacts. A maximum value of 0.5 provides the most protection, but may increase the indexing effect, which may negatively impact sound quality. Reducing the value may decrease the extent of protection, but reduce the indexing effect.')

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            generate_btn.click(song_cover_pipeline,
                              inputs=[local_file, rvc_model, pitch, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, output_format],
                              outputs=[converted_voice])

        with gr.Tab('Model Downloader'):
            with gr.Tab('Download from Link'):
                with gr.Row():
                    with gr.Column(variant='panel'):
                        gr.HTML("<center><h3>Enter a link to the ZIP archive in the field below.</h3></center>")
                        model_zip_link = gr.Text(label='Model Download Link')
                    with gr.Column(variant='panel'):
                        with gr.Group():
                            model_name = gr.Text(label='Model Name', info='Give your uploaded model a unique name, different from other voice models.')
                            download_btn = gr.Button('Download Model', variant='primary')

                gr.HTML("<h3>Supported sites: <a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, <a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a>, <a href='https://drive.google.com/' target='_blank'>Google Drive</a>, <a href='https://mega.nz/' target='_blank'>Mega</a>, <a href='https://disk.yandex.ru/' target='_blank'>Yandex Disk</a></h3>")
                
                dl_output_message = gr.Text(label='Output Message', interactive=False)
                download_btn.click(download_from_url, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            with gr.Tab('Upload a ZIP archive'):
                with gr.Row(equal_height=False):
                    with gr.Column(variant='panel'):
                        zip_file = gr.File(label='Zip File', file_types=['.zip'], file_count='single')
                    with gr.Column(variant='panel'):
                        gr.HTML("<h3>1. Find and download the files: .pth and optional .index file</h3>")
                        gr.HTML("<h3>2. Put the file(s) into a ZIP archive and place it in the upload area</h3>")
                        gr.HTML('<h3>3. Wait for the ZIP archive to fully upload to the interface</h3>')
                        with gr.Group():
                            local_model_name = gr.Text(label='Model Name', info='Give your uploaded model a unique name, different from other voice models.')
                            model_upload_button = gr.Button('Upload Model', variant='primary')

                local_upload_output_message = gr.Text(label='Output Message', interactive=False)
                model_upload_button.click(upload_zip_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

            with gr.Tab('Upload files'):
                with gr.Group():
                    with gr.Row():
                        pth_file = gr.File(label='.pth file', file_types=['.pth'], file_count='single')
                        index_file = gr.File(label='.index file', file_types=['.index'], file_count='single')
                with gr.Column(variant='panel'):
                    with gr.Group():
                        separate_model_name = gr.Text(label='Model Name', info='Give your uploaded model a unique name, different from other voice models.')
                        separate_upload_button = gr.Button('Upload Model', variant='primary')

                separate_upload_output_message = gr.Text(label='Output Message', interactive=False)
                separate_upload_button.click(upload_separate_files, inputs=[pth_file, index_file, separate_model_name], outputs=separate_upload_output_message)

    app.launch(share=True, show_api=False).queue(api_open=False)
