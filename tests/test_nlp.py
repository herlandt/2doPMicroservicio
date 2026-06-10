"""CU-39 — tests del mapeo voz→formulario.

Verifican que, dado un TEXTO dictado, cada entidad (nombre, DNI, fecha,
teléfono, correo) se coloque en el CAMPO que corresponde según su nombre/tipo.
El audio→texto sigue siendo stub (falta Whisper); aquí probamos el mapeo, que
es la parte "rellenarse donde corresponde los campos correspondientes".

Correr:  cd ia_service && .venv/Scripts/python -m pytest tests/ -q
"""
from app.routers.nlp import _sugerir
from app.schemas.nlp import CampoSchema


def s(nombre: str, texto: str, tipo: str = "texto"):
    return _sugerir(CampoSchema(nombre=nombre, tipo=tipo), texto)


# ── DNI / cédula ─────────────────────────────────────────────────────────────

def test_dni_se_extrae():
    r = s("dni", "el cliente tiene DNI 12345678", "numero")
    assert r.valor == "12345678"
    assert r.confianza >= 0.9


def test_cedula_alias_de_dni():
    r = s("cedula_identidad", "su cédula es 7654321")
    assert r.valor == "7654321"


def test_dni_sin_numero_queda_vacio_baja_confianza():
    r = s("dni", "el cliente no recuerda su documento")
    assert r.valor == ""
    assert r.confianza < 0.5


# ── Nombre (ya NO está hardcodeado) ──────────────────────────────────────────

def test_nombre_se_extrae_del_texto():
    r = s("nombre_cliente", "se llama Pedro García López")
    assert r.valor == "Pedro García López"
    assert r.confianza >= 0.8


def test_nombre_no_devuelve_juan_perez_fijo():
    r = s("nombre", "el cliente es Maria Quispe")
    assert r.valor == "Maria Quispe"
    assert "Juan" not in r.valor  # antes era hardcodeado "Juan Perez"


def test_nombre_corta_en_stopword():
    r = s("nombre_completo", "el nombre del cliente es Ana Lopez, su DNI es 9087654")
    assert r.valor == "Ana Lopez"


# ── Fecha ────────────────────────────────────────────────────────────────────

def test_fecha_textual_con_anio():
    r = s("fecha_inspeccion", "la inspección es el 15 de marzo de 2026", "fecha")
    assert r.valor == "15/03/2026"


def test_fecha_numerica():
    r = s("fecha", "programada para 03/05/2026", "fecha")
    assert r.valor == "03/05/2026"


def test_fecha_relativa_manana():
    r = s("fecha_cita", "la cita es mañana a las 10", "fecha")
    assert r.valor == "mañana"


# ── Teléfono / correo ────────────────────────────────────────────────────────

def test_telefono_celular():
    r = s("telefono_contacto", "su celular es 78451234")
    assert r.valor == "78451234"


def test_email():
    r = s("correo", "escribe a juan.perez@mail.com por favor", "email")
    assert r.valor == "juan.perez@mail.com"


# ── Campo sin heurística ─────────────────────────────────────────────────────

def test_campo_desconocido_queda_vacio():
    r = s("color_favorito", "me gusta el color azul")
    assert r.valor == ""
    assert r.confianza < 0.5


# ── EL test clave: cada entidad en SU campo (no se cruzan) ────────────────────

def test_cada_entidad_cae_en_su_campo():
    texto = (
        "El cliente es Ana Lopez, su DNI es 9087654, "
        "la cita el 20 de junio de 2026, correo ana@x.com, celular 71122334"
    )
    assert s("nombre_cliente", texto).valor == "Ana Lopez"
    assert s("dni", texto, "numero").valor == "9087654"
    assert s("fecha_cita", texto, "fecha").valor == "20/06/2026"
    assert s("correo", texto, "email").valor == "ana@x.com"
    assert s("telefono", texto).valor == "71122334"


def test_respuesta_conserva_el_nombre_del_campo():
    # El 'campo' devuelto debe ser EXACTAMENTE el del schema (para aplicarlo al form).
    r = s("Nombre_Del_Cliente", "se llama Luis Mamani")
    assert r.campo == "Nombre_Del_Cliente"


# ── Select / checkbox (CU-39, tipos con opciones / sí-no) ────────────────────

def test_select_coincide_con_una_opcion():
    r = _sugerir(
        CampoSchema(
            nombre="resultado",
            tipo="select",
            opciones=["Conforme", "Con observaciones", "No conforme"],
        ),
        "tras revisar todo, la solicitud quedó Conforme",
    )
    assert r.valor == "Conforme"
    assert r.confianza >= 0.8


def test_select_sin_coincidencia_queda_vacio():
    r = _sugerir(
        CampoSchema(nombre="resultado", tipo="select", opciones=["Aprobado", "Rechazado"]),
        "el cliente trajo sus papeles",
    )
    assert r.valor == ""
    assert r.confianza < 0.5


def test_checkbox_afirmativo():
    r = _sugerir(
        CampoSchema(nombre="documentos_completos", tipo="checkbox"),
        "los documentos están completos y conforme a lo exigido",
    )
    assert r.valor == "true"


def test_checkbox_negativo():
    r = _sugerir(
        CampoSchema(nombre="documentos_completos", tipo="checkbox"),
        "la entrega está incompleta, falta el plano",
    )
    assert r.valor == "false"


def test_etiqueta_aporta_al_matching():
    # El nombre técnico es opaco ('campo_1') pero la etiqueta dice 'Correo'.
    r = _sugerir(
        CampoSchema(nombre="campo_1", tipo="texto", etiqueta="Correo electrónico"),
        "escríbele a ana@x.com",
    )
    assert r.valor == "ana@x.com"
