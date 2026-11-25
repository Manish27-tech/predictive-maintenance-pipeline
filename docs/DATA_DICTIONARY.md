# Data dictionary

## Raw columns

| Column | Type | Description |
| --- | --- | --- |
| UDI | integer | Row index |
| Product ID | string | Product identifier |
| Type | category | Product quality variant: L, M or H |
| Air temperature [K] | numeric | Ambient air temperature |
| Process temperature [K] | numeric | Process temperature |
| Rotational speed [rpm] | numeric | Spindle speed |
| Torque [Nm] | numeric | Torque |
| Tool wear [min] | numeric | Accumulated tool wear |
| Machine failure | binary | Target: failure occurred |
| TWF | binary | Tool-wear failure |
| HDF | binary | Heat-dissipation failure |
| PWF | binary | Power failure |
| OSF | binary | Overstrain failure |
| RNF | binary | Random failure |

## Derived audit columns

| Column | Description |
| --- | --- |
| tool_wear_band | 0-50, 51-120, 121-180, 181-300 minutes |

## Engineered features

| Column | Description |
| --- | --- |
| temp_delta_k | Process temperature minus air temperature |
| power_w | Torque x angular speed |
| strain_index | Torque x tool wear / 1000 |
| torque_per_1000_rpm | Torque divided by speed in thousands of rpm |
| wear_per_power | Tool wear per unit of power |
| temp_ratio | Process/air temperature ratio |
| thermal_load | Temperature delta x torque / 100 |
| speed_deficit | 1500 rpm minus actual speed |
