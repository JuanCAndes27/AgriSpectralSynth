# Librería espectral

Espectros de reflectancia de los materiales que el motor mezcla en cada píxel. Describen **lo que refleja la superficie**, independientemente del sensor. Cómo ve cada sensor estos espectros está en [`sensors/`](../sensors/README.md).

```
spectral_library/
├── vegetation/
│   ├── prosail_presets.yaml   parámetros PROSAIL de cada estado
│   ├── healthy.csv            sana          NDVI ≈ 0.91
│   ├── stressed.csv           estresada     NDVI ≈ 0.78
│   ├── dry.csv                seca          NDVI ≈ 0.47
│   └── dead.csv               muerta        NDVI ≈ 0.24
├── soil/
│   ├── soil_dry.csv
│   ├── soil_wet.csv
│   └── soil_mixed.csv         70 % seco + 30 % húmedo (por defecto)
├── water/  urban/             pendientes
└── metadata/
```

(El NDVI de la lista es el del espectro puro con bandas del Mavic 3M.)

## Formato

Un CSV por material, de 400 a 1000 nm cada 1 nm, con metadatos en comentarios:

```
# name: healthy
# category: vegetation
# model: PROSPECT-D + 4SAIL (prosail 2.0.5)
# parameters: N=1.5, cab=45.0, ...
wavelength_nm,reflectance
400,0.01774
...
```

Se puede agregar cualquier espectro con ese formato, por ejemplo de la librería espectral del USGS o de ECOSTRESS, siempre que cubra 400–1000 nm. Se usa desde la configuración:

```yaml
spectral:
  vegetation: vegetation/stressed
  soil: soil/soil_dry
```

## Cómo se generaron

`python scripts/build_spectral_library.py` (requiere `pip install -e ".[prosail]"`):

- **Vegetación:** PROSAIL (PROSPECT-D para la hoja + 4SAIL para el dosel), con sol a 30° del cenit y sensor en nadir, sobre fondo de suelo. Los parámetros de cada estado (clorofila, agua, materia seca, LAI…) están en `vegetation/prosail_presets.yaml`. **Son plausibles para copas arbóreas pero no están calibrados con mediciones de campo.**
- **Suelo:** los dos espectros de suelo (seco y húmedo) que trae el paquete PROSAIL, y su mezcla.

PROSAIL es una dependencia **opcional** con licencia GPLv3. Aquí solo se guardan los espectros que produce, de modo que el simulador funciona sin instalarlo. Si se necesita un suelo de licencia más clara, se recomienda reemplazar los de `soil/` por espectros del USGS, que son de dominio público.
