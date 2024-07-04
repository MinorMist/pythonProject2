"""ADMET URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
'''
from django.contrib import admin
from django.urls import path
from . import views


from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.download_admet_csv, name='download-admet-csv'),
]
'''

from django.contrib import admin
from django.urls import path
from . import views


from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('sample/', views.sample, name='sample'),
    path('result_page/', views.calculate_new, name='result_page'),
    path('fingerprint_page/', views.calculate_new_fingerprint, name='fingerprint_page'),
    path('druglikeness_page/', views.calculate_new_druglikeness, name='druglikeness_page'),
    path('download/', views.download, name='download'),
    path('download_fingerprint/', views.download_fingerprint, name='download_fingerprint'),
    path('fingerprint/', views.calculate_fingerprint, name='calculate_fingerprint'),
    path('download/morgan/', views.download_morgan_csv, name='download_morgan_csv'),
    path('download/maccs/', views.download_maccs_csv, name='download_maccs_csv'),
    path('download/torsion/', views.download_torsion_csv, name='download_torsion_csv'),
    path('download/rdk/', views.download_rdk_csv, name='download_rdk_csv'),
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
]

