# RVEStressStats — встановлення і запуск в ANSYS Mechanical 2025 R2

Розширення показує в Mechanical статистику локальних напружень з ансамблю RVE
(MatViz3D, `result-10-5.hdf5`) у кожному вузлі макромоделі: квантиль, середнє,
std, min, max, Q(макро) або ймовірність перевищення порогу.

## Що вже перевірено

| Перевірка | Стан |
|---|---|
| CPython `C:\Python313\python.exe` з numpy 2.4, h5py 3.16, matplotlib 3.10, scipy 1.17 | готово |
| `python -m mv3d_stress_stats.node_cli` запускається з будь-якої теки через `PYTHONPATH` | готово |
| Усі виклики API в `main.py` є в документації ACT 2025 R2 (`Ansys.ACT.Interfaces.xml`, `Ansys.ACT.WB1.xml`) | готово |
| Назви контролів XML (`scoping`, `select`, `float`, `text`, `fileopen`, `folderopen`) є в `Ansys.ACT.Core.dll` | готово |
| Колбеки `main.py` виконані в CPython з імітацією Mechanical (тест `tests/test_act_main_mock.py`) | готово |
| Запуск у живій сесії Mechanical | **ще не робився — це ваш крок 6** |

## Крок 1. Скопіювати розширення

Тека для розширень користувача вже існує і порожня:
`%APPDATA%\Ansys\v252\ACT\extensions`

У PowerShell:

```powershell
$src = "D:\temp\diss-code\mv3d-stress_ensemble_statistics\act"
$dst = "$env:APPDATA\Ansys\v252\ACT\extensions"
Copy-Item "$src\RVEStressStats.xml" $dst -Force
Copy-Item "$src\RVEStressStats" $dst -Recurse -Force
Get-ChildItem $dst -Recurse | Select-Object FullName
```

Має вийти:

```
...\extensions\RVEStressStats.xml
...\extensions\RVEStressStats\main.py
...\extensions\RVEStressStats\rve_bridge.py
...\extensions\RVEStressStats\images\rve.png
```

Після кожної зміни коду в репозиторії повторюйте копіювання, а в Workbench
виконайте *Extensions → Reload Extensions* (або перезапустіть Mechanical).

## Крок 2. Увімкнути розширення у Workbench

1. Запустіть Workbench 2025 R2.
2. *Extensions → Manage Extensions…*
3. Знайдіть **RVEStressStats** і поставте галочку *Loaded*.
   Щоб воно вмикалося завжди: правий клік → *Load as Default*.
4. Закрийте вікно.

Якщо розширення немає у списку: *Tools → Options → Extensions → Additional
Extension Folders* і додайте `%APPDATA%\Ansys\v252\ACT\extensions`.

## Крок 3. Підготувати макромодель

- Аналіз **Static Structural**, лінійно-пружний.
- Матеріал макромоделі — гомогенізований матеріал RVE. Для `result-10-5.hdf5`
  ансамбль практично ізотропний (коефіцієнт анізотропії Зінера 1,01), тож
  достатньо **Isotropic Elasticity: E = 138,8 ГПа, ν = 0,331**
  (G = 52,4 ГПа; C11 ≈ 206,6, C12 ≈ 102,3, C44 ≈ 52,4 ГПа).
  Для іншого HDF5 повну матрицю C_eff можна отримати так:
  ```bash
  python -c "from mv3d_stress_stats.localization import LocalizationBasis as B; import numpy as np; np.set_printoptions(precision=4, suppress=True); print(B.from_file_cached('result-10-5.hdf5').effective_stiffness()/1e9, 'GPa')"
  ```
  (порядок компонент X, Y, Z, XY, YZ, XZ, інженерний зсув).
- У *Analysis Settings → Output Controls* має бути увімкнено **Stress** і
  **Strain** (за замовчуванням так).
- Розв'яжіть модель (*Solve*).

## Крок 4. Додати результат

1. У дереві виділіть **Solution** потрібного аналізу.
2. На вкладці/панелі **RVE statistics** натисніть:
   - **RVE stress statistic** — квантиль / середнє / std / min / max / Q(macro);
   - **RVE exceedance probability** — P(Q > поріг).
3. Заповніть властивості в *Details*:

| Властивість | Що ввести |
|---|---|
| Geometry | тіла або вузли, де рахувати (для початку — 1 невелике тіло) |
| RVE ensemble → Basis HDF5 file | `D:\temp\diss-code\mv3d-stress_ensemble_statistics\result-10-5.hdf5` |
| RVE ensemble → Set indices | порожньо = усі 100 RVE (або `0,1,2,3,4` для швидкої перевірки) |
| RVE ensemble → Python executable | `C:\Python313\python.exe` |
| RVE ensemble → Package folder | `D:\temp\diss-code\mv3d-stress_ensemble_statistics` |
| Macro state | **Elastic strain (EPEL)** (рекомендовано) або Stress (S) |
| Quantity | `von_mises`, `principal_1`, `hydrostatic_stress`, … |
| Statistic / Quantile level | напр. `Quantile` і `0.95` |
| Threshold (тільки exceedance) | поріг в одиницях напружень розв'язувача, напр. `80` для МПа |
| Solver stress unit | `Auto` (якщо буде помилка — виберіть явно, напр. `MPa`) |
| Plot p(Q\|node) for node IDs | необов'язково, напр. `125,3401` |

Щоб не вводити все заново, налаштований об'єкт можна дублювати
(правий клік → *Duplicate*) і змінити лише Statistic/Quantity.

## Крок 5. Обчислити

Правий клік на об'єкті → **Evaluate All Results**. Перший запуск будує кеш
базису (~8 с; файл `result-10-5.basis-*.npz` поруч з HDF5), далі — ~0,05 с
на завантаження плюс розрахунок (~20 мс на вузол при 100 RVE).

Робочі файли з'являться в теці розв'язку:
`<project>_files\dp0\SYS\MECH\RVEStressStats\`
- `nodes_<id>_set<n>.csv` — макростан, переданий у Python;
- `stats_<id>_set<n>.csv` — повна статистика для всіх вузлів;
- `plots\node_<id>_<quantity>.png` — графіки p(Q|вузол), якщо задано вузли.

## Крок 6. Перша перевірка (обов'язково)

1. Візьміть модель-куб під **одновісним розтягом** σx = 100 МПа.
2. Додайте RVE stress statistic: Quantity = `von_mises`, Statistic = `Q(macro)`.
   Очікується ≈ 100 МПа в усіх вузлах (Q від гомогенізованого напруження).
3. Statistic = `Mean` → трохи більше (~104–106 МПа),
   `Quantile 0.95` → ≈ 134 МПа, `Quantile 0.05` → ≈ 72 МПа
   (порівняйте з `figures/direction_sweep/sweep_sx_sy_von_mises.png`, θ = 0°).
4. Для того самого вузла порівняйте `stats_*.csv` з
   ```bash
   python -m mv3d_stress_stats.node_cli --basis result-10-5.hdf5 --input <nodes csv з кроку 5> --kind strain --stress-unit MPa --quantity von_mises --probs 0.05,0.5,0.95 --out check.csv
   ```

## Якщо щось не так

| Симптом | Причина / дія |
|---|---|
| Розширення не видно в Manage Extensions | файли не в `%APPDATA%\Ansys\v252\ACT\extensions` або XML лежить усередині теки — XML має бути поруч з текою `RVEStressStats` |
| Немає кнопок RVE statistics | розширення не *Loaded*; відкрийте Mechanical заново |
| `cannot map solver stress unit '…'` | виберіть *Solver stress unit* явно |
| `basis HDF5 file not found` | перевірте шлях у *Basis HDF5 file* |
| `node_cli failed (exit 1)` + текст Python | у тексті причина: `No module named mv3d_stress_stats` → не задано *Package folder*; `No module named numpy/h5py` → вибрано інший python |
| Помилка про `GetNodeValues` / `EPEL` | у файлі результатів немає пружних деформацій — увімкніть Strain у Output Controls або виберіть Macro state = Stress (S) |
| Значення в 10⁶ разів не ті | невідповідність одиниць — задайте *Solver stress unit* явно |

Журнал ACT: *Extensions → View Log File* (туди пишеться stdout Python з
префіксом `[RVEStressStats]`). При помилці надішліть текст помилки і
останні рядки журналу.

## Обмеження

- RVE лінійно-пружні (суперпозиція 6 одиничних випадків точна лише для лінійної задачі).
- Матеріал макромоделі має відповідати гомогенізованому RVE.
- Макростан береться з усереднених у вузлах результатів Mechanical.
