from PIL import Image
import io
from fitz.utils import getColor
import fitz
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from vkr.loader.ct_loader import SIZE
from vkr.settings import MAIL, SMTP_SERVER, MAIL_PSWD
import datetime

blue = getColor("blue")
black = getColor("black")


def send_mail(target, text, result):
    mail = MAIL
    server = smtplib.SMTP(SMTP_SERVER, 587)
    server.ehlo()
    server.starttls()
    server.login(mail, MAIL_PSWD)
    msg = MIMEMultipart()
    msg['From'] = mail
    msg['To'] = target
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = "Результат анализа КТ-снимка от " + str(datetime.datetime.now().strftime('%H:%M %d.%m.%Y'))
    msg.attach(MIMEText(text))

    byte_file = result.convert_to_pdf()
    part = MIMEApplication(
        byte_file,
        Name='report.pdf'
    )
    part['Content-Disposition'] = 'attachment; filename="%s"' % 'report.pdf'
    msg.attach(part)
    server.sendmail(mail, target, msg.as_string())
    server.quit()
    server.close()


def image_to_stream(image):
    rgb = Image.fromarray(image, mode='RGB')
    img_byte_arr = io.BytesIO()
    rgb.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def report_generator(data=None, times=None, f_name='test.zip'):
    doc = fitz.open()
    doc.insertPage(pno=0)
    report_page = doc[-1]
    report_text = fitz.TextWriter(report_page.rect, color=black)
    report_text.append(pos=fitz.Point(36, 32), text="Отчёт к результату анализа КТ-снимка " + str(f_name),
                       language='ru', fontsize=14)

    report_text.append(pos=fitz.Point(400, 52), text="Начато: " + times['start'].strftime('%H:%M %d.%m.%Y'),
                       language='ru', fontsize=12)
    report_text.append(pos=fitz.Point(400, 64), text="Зав-но: " + times['end'].strftime('%H:%M %d.%m.%Y'),
                       language='ru', fontsize=12)

    last_pos = 90
    for line in data['text'].split("\n"):
        report_text.append(pos=fitz.Point(36, last_pos), text=line, language='ru', fontsize=12)
        last_pos = last_pos + 15
    page = 1
    last_pos += 45
    report_text.append(pos=fitz.Point(550, 800), text=str(page), language='ru', fontsize=12)
    report_text.append(pos=fitz.Point(36, last_pos),
                       text="Соответствие цвета выделенного  уплотнения и области лёгкого:",
                       fontsize=12)
    last_pos += 15
    writers = [report_text]
    for lobe in data['lobes']:
        report_text = fitz.TextWriter(report_page.rect, color=data['lobes'][lobe].get_fitz_color())
        report_text.append(pos=fitz.Point(36, last_pos), text=data['lobes'][lobe].get_full_label(), fontsize=12)
        last_pos += 15
        writers.append(report_text)
    report_page.writeText(writers=writers)
    report_page.drawLine(fitz.Point(36, 40), fitz.Point(550, 40), color=blue)

    doc.insertPage(pno=page)
    cur_page = doc[-1]
    cur_writer = fitz.TextWriter(cur_page.rect, color=black)
    cur_writer.append(pos=fitz.Point(36, 32), text="Перечень срезов с найденными уплотнениями", language='ru',
                      fontsize=14)
    cur_page.drawLine(fitz.Point(36, 40), fitz.Point(550, 40), color=blue)

    page = page + 1
    pos_x = 36
    pos_y = 60
    total_len = len(data['images'])
    for i in range(data['start'], data['end'] + 1):
        if i >= len(data['images']):
            break
        sl = data['images'][i]
        if cur_page.rect.height <= pos_y + 36 + SIZE:
            cur_writer.append(pos=fitz.Point(550, 800), text=str(page), language='ru', fontsize=12)
            cur_page.writeText(writers=cur_writer)
            doc.insertPage(pno=page)
            cur_page = doc[-1]
            page = page + 1
            cur_writer = fitz.TextWriter(cur_page.rect, color=black)
            pos_x = 36
            pos_y = 60
            cur_writer.append(pos=fitz.Point(36, 32), text="Перечень срезов с найденными уплотнениями", language='ru',
                              fontsize=14)
            cur_page.drawLine(fitz.Point(36, 40), fitz.Point(550, 40), color=blue)
        if cur_page.rect.width <= pos_x + SIZE + 5:
            if cur_page.rect.height <= pos_y + 24 * 2 + SIZE * 2:
                cur_writer.append(pos=fitz.Point(550, 800), text=str(page), language='ru', fontsize=12)
                cur_page.writeText(writers=cur_writer)
                doc.insertPage(pno=page)
                cur_page = doc[-1]
                page = page + 1
                cur_writer = fitz.TextWriter(cur_page.rect, color=black)
                pos_x = 36
                pos_y = 60
                cur_writer.append(pos=fitz.Point(36, 32), text="Перечень срезов с найденными уплотнениями",
                                  language='ru', fontsize=14)
                cur_page.drawLine(fitz.Point(36, 40), fitz.Point(550, 40), color=blue)
            else:
                pos_y = pos_y + 16 + SIZE
                pos_x = 36
        cur_writer.append(pos=fitz.Point(pos_x, pos_y), text=str(i + 1) + '/' + str(total_len), language='ru',
                          fontsize=12)
        cur_page.insertImage(rect=fitz.Rect(pos_x, pos_y + 4, pos_x + SIZE, pos_y + SIZE + 4),
                             stream=image_to_stream(sl))
        pos_x = pos_x + SIZE + 5
    cur_writer.append(pos=fitz.Point(550, 800), text=str(page), language='ru', fontsize=12)
    cur_page.writeText(writers=cur_writer)

    return doc


def send_report(times, result, mail, f_name):
    doc = report_generator(result, times, f_name=f_name)
    send_mail(mail, result['text'], doc)
