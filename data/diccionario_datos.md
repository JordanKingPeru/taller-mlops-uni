# Diccionario de datos — AndesTel (caso del taller)

Dataset sintético de clientes de una empresa de telecomunicaciones peruana ficticia (**AndesTel**), creado para el taller *MLOps en la práctica* (UNI). Las relaciones entre variables son realistas pero los datos no corresponden a ninguna persona real.

## Archivos

| Archivo | Filas | Uso en el taller |
|---|---|---|
| `churn_telco_peru.csv` | 7,560 | Entrenamiento y experimentación (Labs 1–3). Contiene nulos y duplicados **a propósito**. |
| `churn_telco_peru_nuevos.csv` | 1,500 | Lote "de producción" (periodo 2026) con **data drift inyectado**. Se usa en el Lab 5 de monitoreo. |

## Variables

| Columna | Tipo | Descripción |
|---|---|---|
| `id_cliente` | str | Identificador único del cliente (no usar como feature). |
| `edad` | float | Edad del titular (18–85). Tiene ~2% de nulos. |
| `departamento` | str | Departamento del Perú donde reside (8 categorías, Lima ≈ 42%). |
| `plan` | str | Prepago, Postpago Básico, Postpago Plus, Postpago Premium. |
| `tipo_contrato` | str | Mensual, Anual, 18 meses. Prepago siempre es Mensual. |
| `meses_antiguedad` | int | Meses como cliente (1–96). |
| `cargo_mensual_soles` | float | Facturación mensual promedio en soles (S/). |
| `gb_datos_mes` | float | GB de datos consumidos al mes. Tiene ~3% de nulos. |
| `minutos_llamadas_mes` | int | Minutos de voz al mes. |
| `lineas_adicionales` | int | Líneas extra en la misma cuenta (0–3). |
| `tickets_soporte_6m` | int | Reclamos/tickets de soporte en los últimos 6 meses. |
| `caidas_servicio_mes` | int | Caídas de servicio reportadas en su zona por mes. |
| `dias_ultimo_pago_vencido` | int | Días de atraso del último pago (0 = pagó a tiempo). |
| `factura_electronica` | int | 1 si usa factura electrónica, 0 si física. |
| `churn` | int | **Target.** 1 = el cliente se dio de baja en el siguiente trimestre. Tasa ≈ 13.9% en train. |

## Señales que el modelo debería aprender (para discusión en clase)

Mayor churn asociado a: poca antigüedad, muchos tickets de soporte, caídas de servicio, pagos atrasados, plan Prepago y pagar más de lo típico para su plan. Menor churn: contratos anuales/18 meses, líneas adicionales y factura electrónica.

## Drift inyectado en el lote 2026 (`churn_telco_peru_nuevos.csv`)

Pensado para que Evidently lo detecte en el Lab 5: más clientes nuevos (campaña de captación → cae `meses_antiguedad`), subida de tarifas (+18% en `cargo_mensual_soles`), red degradada (sube `caidas_servicio_mes`) y mayor proporción de Prepago. Consecuencia: la tasa real de churn sube a ≈ 23% y el modelo entrenado con 2025 se degrada.

## Regeneración

El dataset es 100% reproducible: `python generar_dataset.py` (seed fija = 42).
