from django.shortcuts import render, redirect
from .forms import UploadImageForm
from .models import UploadedImage
import folium
from .pre import *
from .direction import *

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            image_path = uploaded_image.image.path
            result = predict(image_path)
            mymap = direction(get_geolocation(), result)
            map_html = mymap._repr_html_()

            return render(request, 'prapp/map.html', {'map_html': map_html})
    else:
        form = UploadImageForm()

    return render(request, 'prapp/upload_image.html', {'form': form})

def success(request):
    return render(request, 'prapp/success.html')
