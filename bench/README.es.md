# Benchmarks

*[English](README.md) · **Español***

El repositorio hace varias afirmaciones de rendimiento. Este proyecto las mide, incluida la que
resultó **errónea**.

```bash
dotnet run -c Release --project bench/NN.Bench -- --filter '*'
dotnet run -c Release --project bench/NN.Bench -- --filter '*DotProduct*'   # un grupo
```

La configuración `Release` es obligatoria: BenchmarkDotNet se niega a ejecutarse con cualquier otra,
y una compilación `Debug` no mediría nada significativo.

> **Sobre el idioma.** Los nombres de los benchmarks, las clases y las columnas de las tablas
> proceden del código y de la salida de BenchmarkDotNet, así que se conservan en inglés. Traducirlos
> describiría un proyecto que no existe.

## La máquina de la que salen estos números

> Apple M3 Pro (11 núcleos), macOS 26.3, **.NET 10.0.10**, Arm64 RyuJIT armv8.0-a, BenchmarkDotNet
> 0.15.8. Todas las secciones usan el `job` por defecto de BenchmarkDotNet; en la nota de la
> sección 1 se explica por qué `--job short` no basta para sacar una conclusión.

**Aquí `Vector<float>` tiene ancho 4.** En una máquina x86 con AVX2 son 8, y con AVX-512 son 16, así
que las proporciones SIMD de abajo cambiarán; la *forma* de cada resultado no debería cambiar.
Vuelve a ejecutarlo en tu propio hardware: para eso se publica el proyecto y no solo la tabla.

## 1. SIMD y el segundo acumulador

`SimdOps.Dot` frente al bucle escalar evidente, y frente a una versión vectorizada con un solo
acumulador.

| Longitud | Escalar | 1 acumulador | 2 acumuladores (el que se publica) | Ganancia SIMD | 2.º acumulador |
|---|---|---|---|---|---|
| 8 | 4,23 ns | 1,27 ns | 1,03 ns | **4,1×** | **1,23×** |
| 64 | 38,3 ns | 9,92 ns | 7,91 ns | **4,8×** | **1,25×** |
| 512 | 360 ns | 82,4 ns | 63,5 ns | **5,7×** | **1,30×** |
| 4096 | 2937 ns | 723 ns | 500 ns | **5,9×** | **1,45×** |

**Ambas afirmaciones se sostienen.** Vectorizar aporta entre 4,1× y 5,9× en cuanto hay suficiente
trabajo para amortizar la preparación del bucle, y el segundo acumulador añade otro 1,2–1,5×: más de
lo que sugeriría el vector de ancho 4 por sí solo, que es justamente el efecto de segmentación
(`pipelining`) que describe el comentario del código.

> **Estas filas se midieron con `--job short` en una revisión anterior y decían otra cosa.** La
> longitud 8, en particular, mostraba que el segundo acumulador *perdía* un 17 %, y el texto de aquí
> explicaba por qué a esa longitud debía perder. Con el `job` por defecto gana 1,23×. El margen de
> error del `job` corto, de ±0,26 ns, era sencillamente más ancho que el efecto que se pretendía
> describir. La lección merece conservarse aunque el hallazgo no sobreviviera: **un `job` de tres
> iteraciones sirve para triaje, no para conclusiones.**

### ¿Por qué no llamar directamente a `TensorPrimitives.Dot`?

Porque aquí pierde, que no es lo que uno supondría. `System.Numerics.Tensors` incluye `kernels`
ajustados a mano, y el movimiento obvio es borrar el bucle de arriba y llamar a uno de ellos.

| Longitud | 1 acumulador | 2 acumuladores (el que se publica) | `TensorPrimitives.Dot` |
|---|---|---|---|
| 8 | 1,27 ns | **1,03 ns** | 2,35 ns (**2,3× más lento**) |
| 64 | 9,92 ns | 7,91 ns | **5,98 ns** (1,32× más rápido) |
| 512 | 82,4 ns | **63,5 ns** | 61,4 ns (empate) |
| 4096 | 723 ns | **500 ns** | 734 ns (**1,47× más lento**) |

Fíjate en la fila de 4096: `TensorPrimitives.Dot` queda a menos del 2 % de la versión de un *único*
acumulador. Un producto escalar termina en una reducción, y el `kernel` arrastra una sola cadena de
acumulación a través de ella: exactamente la dependencia serie que el segundo acumulador existe para
romper. En la longitud 8 también pierde con claridad, porque ahí es una llamada real mientras que
`SimdOps.Dot` se inserta en línea, y la llamada cuesta más que la aritmética. Por eso `Dot` sigue
escrito a mano.

Con `AddScaled` ocurre justo lo contrario —véase la sección 5—, y por eso la biblioteca usa
`TensorPrimitives` para uno de sus dos primitivos y no para el otro. Ninguna de las dos decisiones
era predecible leyendo la API; ambas salieron de esta tabla.

## 2. Disposición de los pesos: por unidad frente a por característica

Mismos pesos, misma aritmética, misma activación. La única diferencia es el orden en memoria: por
unidad (contiguo, lo que se publica) frente a la disposición por característica `(inputs, units)` de
NumPy (con salto).

| Forma de la capa | Por unidad | Por característica | Coste del salto |
|---|---|---|---|
| 2 × 4 (la capa XOR) | 19,1 ns | 16,0 ns | **0,84× — el salto es *más rápido*** |
| 64 × 64 | 525 ns | 3249 ns | **6,2×** |
| 784 × 128 (tamaño MNIST) | 9,75 µs | 72,6 µs | **7,4×** |

**La afirmación se sostiene, con un matiz que la documentación antes omitía.** A tamaños realistas
la disposición contigua vale entre 6,2× y 7,4×, el mayor efecto medido aquí, lo que justifica
llamarla la decisión de diseño más determinante.

Pero en 2×4 pierde. Ocho pesos caben dentro de una línea de caché de 64 bytes, así que no hay línea
de caché que desaprovechar ni acceso disperso que evitar: solo queda la preparación adicional del
camino SIMD, que el bucle escalar con salto se ahorra. **La demostración de XOR es precisamente el
tamaño al que nada de esto importa.** Merece decirse con claridad, porque es el primer ejemplo que
encuentra quien lee.

## 3. Activación genérica frente a delegado: la afirmación que era falsa

El README afirmaba que un campo `Func<float, float>` «costaría una llamada indirecta por unidad que
no se puede insertar en línea». No se puede insertar en línea, en efecto. Y no cuesta casi nada.

Fíjate en la columna de control. `Dense<Tanh>` ahora activa una capa entera por llamada (sección 6),
mientras que la versión con delegado sigue activando unidad por unidad, así que esas dos difieren en
algo más que el despacho y no pueden responder a la pregunta sobre el despacho.
`ScalarActivation` es el control honesto: idéntico a la versión con delegado en todo *salvo* en que
su activación es un parámetro de tipo genérico en lugar de un campo `Func`.

| Forma de la capa | Genérico, por unidad | Delegado, por unidad | Coste del despacho | (`Dense<Tanh>` publicado) |
|---|---|---|---|---|
| 2 × 4 | 15,4 ns | 18,3 ns | **1,19×** | 19,1 ns |
| 64 × 64 | 752 ns | 741 ns | **0,99×** | 525 ns |
| 784 × 128 | 12,71 µs | 13,16 µs | **1,04×** | 9,75 µs |

**Coste medido: dentro de ±4 % en todos los tamaños realistas, y el signo ni siquiera es
consistente.** El único caso atípico —1,19× en la capa de 2×4 con tanh— son cuatro unidades de
trabajo que suman unos 15 ns, donde un par de nanosegundos de sobrecarga de llamada todavía es una
fracción visible. Y es además el tamaño de capa al que el rendimiento no importa.

La primera sospecha fue que tanh, una función trascendente que cuesta decenas de ciclos, estuviera
ocultando la llamada, así que el benchmark repite la comparación con ReLU, que es una comparación y
una selección. El resultado apenas se mueve. La razón es aritmética, no de despacho: la activación
se ejecuta **una vez por unidad**, mientras que el producto escalar que la alimenta ejecuta `Inputs`
multiplicaciones-sumas por unidad. Con 784 entradas, la llamada indirecta se amortiza entre 784
multiplicaciones-sumas. Es invisible porque es infrecuente, no porque sea rápida.

El diseño genérico sigue siendo la mejor opción por defecto —se compone sin coste con activaciones
`readonly struct` y deja libre al JIT para insertar en línea—, pero debe justificarse como una
ventaja de *seguridad de tipos y composición*, no de rendimiento. La documentación ya lo dice así.

## 4. `ForwardBatch`: un resultado nulo deliberado

El §25 de la guía de estudio afirma que hoy `ForwardBatch` no aporta nada, porque se limita a
recorrer en bucle el camino de un solo ejemplo. Comprobar las propias afirmaciones negativas es la
única forma de que sigan siendo ciertas.

| Tamaño de lote | Uno a uno | `ForwardBatch` | Proporción |
|---|---|---|---|
| 1 | 1,87 µs | 1,86 µs | 0,99× |
| 32 | 57,9 µs | 58,1 µs | 1,00× |
| 256 | 481 µs | 469 µs | 0,97× |

**Confirmado: ningún beneficio.** Todas las filas quedan dentro del 3 % de la paridad, muy dentro de
la dispersión entre ejecuciones. Aquí no hay ningún efecto real en ninguna de las dos direcciones,
que es exactamente lo que se afirmaba.

(En términos absolutos estas filas son entre 1,4× y 1,6× más rápidas que en la revisión anterior,
por el motivo que explica la sección 6. Ambas columnas se movieron a la vez, así que la proporción
—lo único que esta sección afirma— no cambia.)

Un GEMM por bloques de verdad —reutilizando cada bloque de pesos ya cargado entre muchos ejemplos en
lugar de recorrer la matriz entera por ejemplo— es donde está la ganancia del procesamiento por
lotes, y sigue sin implementarse. Véase el §25, punto 1, y el ejercicio 14 de la guía de estudio.

## 5. `AddScaled`: donde `TensorPrimitives` sí gana

`dest += src * scale`, el caballo de batalla del `backward pass`: se ejecuta dos veces por unidad,
acumulando el gradiente de los pesos y propagando el gradiente hacia la entrada, así que carga con
más trabajo de entrenamiento que `Dot`.

| Longitud | `Vector<float>` a mano | `TensorPrimitives.MultiplyAdd` | Proporción |
|---|---|---|---|
| 8 | **1,29 ns** | 2,84 ns | **2,19× más lento** |
| 64 | 9,87 ns | **8,66 ns** | 1,14× más rápido |
| 512 | 74,6 ns | **29,6 ns** | **2,52× más rápido** |
| 4096 | 566 ns | **222 ns** | **2,55× más rápido** |

**2,5× en cualquier longitud que merezca vectorizarse, así que este sí se publica.** El resultado
opuesto al de `Dot`, con la misma biblioteca, por una razón estructural: esta es una operación de
flujo puro, sin reducción, así que no hay ninguna cadena de dependencias que arrastrar ni nada que
impida al `kernel` desenrollar el bucle tanto como quiera. Además emite una multiplicación-suma
fusionada real, mientras que `dest[i] += src[i] * scale` se compila como una multiplicación y una
suma separadas.

La fila de longitud 8 pierde por el mismo motivo que en `Dot` —una llamada sin insertar en línea
alrededor de tres instrucciones vectoriales— y por el mismo motivo da igual: las longitudes que
llegan hasta aquí son anchos de capa.

## 6. Activar una capa entera de una vez, en lugar de unidad por unidad

`exp` y `tanh` cuestan decenas de ciclos por llamada, y una capa hace una llamada por unidad.
Aplicar la activación a todo el vector de salida a la vez, después de los productos escalares en
lugar de dentro de ellos, permite que `TensorPrimitives` procese cuatro de golpe. Medido sobre la
activación sola, dejando los productos escalares fuera del cuadro:

| Ancho | Sigmoid escalar → vectorial | Tanh escalar → vectorial | Softmax escalar → vectorial |
|---|---|---|---|
| 10 | 16,6 → 16,0 ns (1,04×) | 19,9 → 18,2 ns (1,09×) | 24,3 → 34,6 ns (**0,70×**) |
| 128 | 208 → 92,8 ns (**2,24×**) | 242 → 121 ns (**2,00×**) | 289 → 134 ns (**2,15×**) |
| 1024 | 1643 → 749 ns (**2,21×**) | 1914 → 956 ns (**2,00×**) | 2219 → 1070 ns (**2,07×**) |

**Sigmoid y tanh se publican vectorizadas; `softmax` no.** La columna del ancho explica por qué. Una
capa oculta es tan ancha como quieras hacerla, así que 128 y 1024 son los casos normales y ese 2× es
real. Pero una capa `softmax` tiene *una unidad por clase* —diez, en MNIST—, y con diez pierde 1,4×,
porque la forma numéricamente estable necesita una pasada de resta del máximo antes del `kernel`, y
cuatro pasadas vectorizadas cortas solo superan a tres pasadas escalares cortas cuando las pasadas
dejan de ser cortas. Solo se adelanta a partir de un centenar de clases, aproximadamente. Por eso
`SoftmaxCrossEntropy.Transform` sigue siendo un bucle escalar.

En cualquier caso, `TensorPrimitives.SoftMax` no puede usarse directamente: calcula `exp(z)/Σexp(z)`
de forma literal, sin restar el máximo, así que devuelve `NaN` para `logits` por encima de ~88. Dos
pruebas unitarias ya existentes lo detectaron de inmediato.

### La regresión de 6× que provocó este cambio, y su solución de una palabra

Llevar los dos cambios a `Dense.Forward` —calcular todas las preactivaciones y activar después el
vector— dejó la capa de 784×128 **6,3× más lenta**: de 12,79 µs a 80 µs. La aritmética era idéntica,
y ese mismo código medía 10,7 µs cuando se llamaba directamente desde un benchmark.

El desensamblado mostró la razón. Insertado en línea dentro de `Dense<TActivation>.Forward` —que es
genérico y que para entonces contenía además un `Dot` y una activación vectorizada también
insertados en línea—, el JIT dejaba de eliminar las comprobaciones de límites en las cargas
vectoriales internas, y quedaba una bifurcación de comprobación de rango protegiendo cada `ldr q`:

```asm
ldr     q18, [x22]
cmp     x21, x11
bhi     G_M000_IG35      ← por vector, por iteración
```

La solución es `[MethodImpl(MethodImplOptions.NoInlining)]` sobre `SimdOps.MatVec`. Compilado por su
cuenta —pequeño y no genérico—, se optimiza sin problemas y la capa paga una llamada ordinaria. La
capa final queda en **9,75 µs, 1,31× más rápida que antes de todo esto**, y `Dense<ReLU>` ganó ese
mismo 1,29× sin ningún cambio en su activación: el bucle fusionado original llevaba todo ese tiempo
perdiendo esa misma cantidad por la misma penalización de inserción en línea.

Merece decirse sin rodeos, porque es lo más útil de este archivo: **una optimización local hizo seis
veces más lento justo aquello que optimizaba, y solo el benchmark lo detectó.** Nada en el código
fuente lo sugería. Vuelve a medir `LayerBenchmarks` antes de tocar `MatVec`.

## 7. ¿Se nota algo de esto en el entrenamiento?

Las secciones 1 a 6 miden primitivos y `forward passes`. El entrenamiento es lo que la biblioteca
hace realmente con su tiempo, y un primitivo que gana 2,5× aislado no ha ganado nada hasta que se
pesa frente a todo lo que comparte paso con él. Un `mini-batch` —forward, backward y la
actualización del descenso— sobre `784 → 128 tanh → 10 softmax`, con lote de 32:

| | Con `TensorPrimitives` | `AddScaled` a mano | |
|---|---|---|---|
| **Paso completo** | **607,5 µs** | 815,0 µs | **1,34× más rápido** |
| Solo forward *(control)* | 318,9 µs | 317,9 µs | 1,00× — sin cambio |
| **Mitad backward** *(por resta)* | **288,6 µs** | 497,1 µs | **1,72× más rápido** |

**El paquete se gana su sitio: un tercio menos en cada paso de entrenamiento.** La fila de control es
lo que hace fiable ese resultado. `AddScaled` no aparece en ninguna parte del `forward pass`, así que
si los dos números de forward hubieran divergido, se habría movido algo distinto del cambio previsto
y toda la comparación quedaría en entredicho. Difieren un 0,3 %, dentro del ruido, lo que sitúa la
totalidad de los 207 µs de diferencia en el `backward pass`: exactamente donde vive el primitivo.

**`Backpropagation` es donde esta biblioteca gasta su tiempo, y la sección 5 se quedaba corta.** La
mitad backward es el 48 % del paso, y ejecuta `AddScaled` dos veces por unidad frente a una sola vez
`Dot`, así que el primitivo pesa más en el entrenamiento de lo que puede mostrar cualquier benchmark
del `forward pass`. Por eso un 2,5× en un microbenchmark se convirtió en 1,72× sobre la pasada y
1,34× sobre el paso completo, en lugar del error de redondeo que habría predicho una visión centrada
solo en el forward.

También resuelve una cuestión que las secciones 1 a 6 no podían responder: renunciar a la dependencia
para mantener la biblioteca sin paquetes externos costaría un tercio de su rendimiento de
entrenamiento.

## Qué no se mide aquí

- **Nada multihilo**: la biblioteca es de un solo hilo de principio a fin.
- **GEMM por lotes**, la única gran ganancia que queda; véase la sección 4 y el §25 de la guía de
  estudio.
- **Tiempos de época completa o de ejecución completa**, incluidas la carga de datos y el barajado.
  La sección 7 mide un `mini-batch` aislado; los 37 s de extremo a extremo de la demostración de
  MNIST se reportan en el README, no aquí.

La CI compila este proyecto en cada `push`, de modo que no puede pudrirse, pero no lo ejecuta: los
tiempos de benchmark medidos en runners compartidos en la nube no valen los minutos que cuestan.
