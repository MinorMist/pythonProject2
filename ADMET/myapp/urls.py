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

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.sample, name='sample'),
    path('result_page/', views.calculate_new, name='result_page'),
    path('fingerprint_page/', views.calculate_new_fingerprint, name='fingerprint_page'),
    #path('druglikeness_page/', views.calculate_new_druglikeness, name='druglikeness_page'),
    path('download/', views.download, name='download'),
    path('fingerprint_page/morgan_new/', views.download_new_morgan_csv, name='download_new_morgan_csv'),
    path('fingerprint_page/morgan2048_new/', views.download_new_morgan2048_csv, name='download_new_morgan2048_csv'),
    path('fingerprint_page/maccs_new/', views.download_new_maccs_csv, name='download_new_maccs_csv'),
    path('fingerprint_page/torsion_new/', views.download_new_torsion_csv, name='download_new_torsion_csv'),
    path('fingerprint_page/rdk_new/', views.download_new_rdk_csv, name='download_new_rdk_csv'),
    path('fingerprint_page/avalon_new/', views.download_new_avalon_csv, name='download_new_avalon_csv'),
    path('fingerprint_page/atom_pair_new/', views.download_new_atom_pair_csv, name='download_new_atom_pair_csv'),
    path('druglikeness_page/', views.predict, name='predict'),
]

