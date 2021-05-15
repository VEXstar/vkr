import threading
import vkr.loader.ct_loader as loader
import vkr.recognizers.phase_one as po
import vkr.recognizers.phase_two as pt
from vkr.worker.mail_sender import send_report
import datetime


def thread_worker(path, is_dir, f_name, mail):
    file_name = f_name
    time_start = datetime.datetime.now()
    data_loader, dataset = loader.get_dataloader(path, is_dir)
    print("for user " + str(mail) + " loaded " + str(len(dataset)) + " batches")
    masks = po.predict(data_loader)
    print("for user " + str(mail) + " complete prediction, total masks: " + str(len(masks)))
    text, images = pt.predict(data_loader, dataset, masks)
    times = {
        'start': time_start,
        'end': datetime.datetime.now()
    }
    send_report(times, images, text, mail, file_name)
    print("for user " + str(mail) + " generated and sent report")


def do_work_user_ct(file, mail):
    file_name = file.name
    path, is_dir = loader.handle_uploaded_file(file)
    t = threading.Thread(target=thread_worker, args=(path, is_dir, file_name, mail), daemon=True)
    t.start()
    return True
