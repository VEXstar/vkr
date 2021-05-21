from django.http import HttpResponse
from vkr.worker.analizator import do_work_user_ct


def index(request):
    if request.method == 'POST':
        if 'mail' not in request.POST:
            return HttpResponse("{\"status\":\"mail not found\"}")
        if len(request.FILES) > 0:
            if 'ct_file' not in request.POST:
                return HttpResponse("{\"status\":\"ct_file not found\"}")
            f = request.FILES['ct_file']
            do_work_user_ct(f, request.POST['mail'])
            return HttpResponse("{\"status\":\"ok\",\"fname\":\"" + f.name + "\"}")
        else:
            return HttpResponse("{\"status\":\"empty body\"}")
    else:
        return HttpResponse("{\"status\":\"not_post\"}")
