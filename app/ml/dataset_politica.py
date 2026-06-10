"""Generador de dataset SINTÉTICO para el clasificador de política (CU-40).

DOMINIO-AGNÓSTICO: el sistema es una "fábrica de trámites" para CUALQUIER
negocio (no solo eléctrico). Por eso el dataset NO contiene vocabulario de
ningún dominio: se genera ÚNICAMENTE a partir del texto que el admin escribió en
cada política (nombre + descripción + categoría), combinándolo con plantillas
genéricas de cómo un cliente formula una solicitud. Si mañana el negocio es un
restaurante o una clínica, el mismo generador produce el dataset a partir de las
políticas de ESE negocio.

Como las políticas son DINÁMICAS, el modelo se RE-ENTRENA cuando cambian
(P2 §3.2.2: "cuando se crean nuevas políticas, el modelo debe poder reentrenarse
o ajustarse para reconocerlas").

Recomendación para el admin: cuanto más descriptivo sea el texto de la política
(incluyendo palabras que el cliente usaría), mejor clasifica el modelo — porque
de ese texto sale el entrenamiento.
"""
import re
import unicodedata

# Palabras vacías genéricas (no de dominio): no aportan señal para distinguir
# una política de otra.
_STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "u",
    "a", "en", "para", "por", "con", "del", "al", "que", "se", "su", "sus", "mi",
    "tu", "lo", "le", "es", "son", "como", "proceso", "solicitud", "tramite",
    "nuevo", "nueva", "este", "esta", "ante", "tras", "sobre", "mediante",
}

# Plantillas genéricas: cómo un cliente describe lo que necesita ({t} = tema
# tomado del texto de la política). Sin ninguna palabra de dominio.
_PLANTILLAS = [
    "{t}",
    "necesito {t}",
    "quiero {t}",
    "quisiera {t}",
    "solicito {t}",
    "quiero hacer el tramite de {t}",
    "necesito el tramite de {t}",
    "como hago para {t}",
    "necesito ayuda con {t}",
    "deseo solicitar {t}",
    "tengo que hacer {t}",
    "vengo por {t}",
    "vengo a tramitar {t}",
    "me gustaria {t}",
    "consulta sobre {t}",
    "para {t}",
    "{t} por favor",
]


def _norm(texto: str) -> str:
    """minúsculas y sin acentos; colapsa espacios."""
    t = unicodedata.normalize("NFD", texto or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).strip().lower()


def _palabras_contenido(texto: str) -> list[str]:
    """Palabras significativas (sin stopwords, len>2), en orden y sin repetir."""
    vistas: list[str] = []
    for w in re.findall(r"[a-z]+", _norm(texto)):
        if len(w) > 2 and w not in _STOPWORDS and w not in vistas:
            vistas.append(w)
    return vistas


def frases_de_politica(nombre: str, descripcion: str, categoria: str) -> list[str]:
    """Genera las frases de entrenamiento para UNA política, solo desde su texto."""
    base = f"{nombre} {descripcion} {categoria}"
    palabras = _palabras_contenido(base)

    # Temas: nombre completo + categoría + palabras de contenido + bigramas.
    # Todo proviene del texto de la política → cero dependencia de dominio.
    temas: list[str] = []
    nom = _norm(nombre)
    cat = _norm(categoria)
    if nom:
        temas.append(nom)
    if cat and cat != nom:
        temas.append(cat)
    temas.extend(palabras[:10])
    for i in range(min(4, len(palabras) - 1)):
        temas.append(f"{palabras[i]} {palabras[i + 1]}")

    frases: set[str] = {nom, _norm(descripcion)}
    for tema in temas:
        if not tema:
            continue
        for tpl in _PLANTILLAS:
            frases.add(tpl.format(t=tema))

    return [f for f in frases if f]


def generar_dataset(politicas: list[dict]) -> tuple[list[str], list[str]]:
    """Devuelve (textos, labels) donde label = id de la política. ``politicas``:
    lista de dicts con id/nombre/descripcion/categoria."""
    textos: list[str] = []
    labels: list[str] = []
    for p in politicas:
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        for f in frases_de_politica(
            str(p.get("nombre") or ""),
            str(p.get("descripcion") or ""),
            str(p.get("categoria") or ""),
        ):
            textos.append(f)
            labels.append(pid)
    return textos, labels


# Políticas semilla SOLO para entrenar/demostrar por CLI y en tests sin depender
# del backend (coinciden con el PoliticaSeeder). Son DATOS de ejemplo, no lógica
# de dominio: el entrenamiento real lo dispara el backend con SUS políticas
# activas — sean del negocio que sean — vía POST /asignacion/reentrenar.
SEED_POLITICAS: list[dict] = [
    {"id": "pol-conexion", "nombre": "Nueva conexion residencial",
     "descripcion": "Proceso de solicitud de nueva conexion electrica residencial",
     "categoria": "conexiones"},
    {"id": "pol-reconexion", "nombre": "Reconexion por mora",
     "descripcion": "Proceso de reconexion del servicio electrico tras pago de deuda",
     "categoria": "reconexiones"},
    {"id": "pol-titular", "nombre": "Cambio de titular",
     "descripcion": "Proceso administrativo para transferir la titularidad de un contrato de servicio",
     "categoria": "administrativo"},
]
