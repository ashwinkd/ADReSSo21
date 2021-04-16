from global_var import *

language = "task_language=en"
text_type = "is_text_type=plain"
output_format = "os_task_file_format=json"
smil_audio_file = "os_task_file_smil_audio_ref={}"
smil_txt_file = "os_task_file_smil_page_ref={}"


def align(audio_file, transcript):
    file_id = audio_file.split('/')[-1].split('.')[0]
    text_file = os.path.join("tmp", "text", "{}.txt".format(file_id))
    out_file = os.path.join("tmp", "out", "{}.json".format(file_id))
    CONFIG = "|".join([language,
                       text_type,
                       output_format])
    # smil_audio_file.format(audio_file),
    # smil_txt_file.format(text_file)])
    words = transcript.split()
    plain_text = "\n".join(words)
    with open(text_file, 'w') as fptr:
        fptr.write(plain_text)

    command = 'python -m aeneas.tools.execute_task {} {} "{}" {}'.format(audio_file, text_file, CONFIG, out_file)
    print(command)
    os.system(command)