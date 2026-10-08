# Curvas de respuesta espectral medidas (opcional)

Coloca aquí archivos CSV con la curva real de una banda para reemplazar la
aproximación gaussiana. Formato (dos columnas, con encabezado):

```
wavelength_nm,response
650,0.01
651,0.03
...
```

Y en el YAML del sensor:

```yaml
bands:
  - {name: B4, center: 664.6, fwhm: 31, srf_file: srf/sentinel2a_B4.csv}
```

La ruta es relativa a la carpeta `sensors/`. La respuesta se normaliza sola.
