from django.http import HttpResponse
from vkr.worker.analizator import do_work_user_ct


def index(request):
    if request.method == 'POST':
        if len(request.FILES) > 0:
            f = request.FILES['ct_file']
            do_work_user_ct(f, request.POST['mail'])
            return HttpResponse("{\"status\":\"ok\",\"fname\":\"" + f.name + "\"}")
        else:
            return HttpResponse("{\"status\":\"empty body\"}")
    else:
        return HttpResponse("{\"status\":\"not_post\"}")

# def test(request):
#     if request.method == 'POST':
#         if len(request.FILES) > 0:
#             f = request.FILES['ct_file']
#             path = handle_uploaded_file(f)
#             elem = ua.deco_do_analyzing(path)
#             return render(request, "plot_view.html",
#                           {'ct': elem['scan'], 'mask': elem['mask'], 'range': range(len(elem['mask']))})
#         else:
#             return HttpResponse("{\"status\":\"empty body\"}")
#     else:
#         return HttpResponse("{\"status\":\"not_post\"}")
