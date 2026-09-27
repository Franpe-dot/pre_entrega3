"""Modelos Pydantic de la respuesta del sistema RAG."""

from pydantic import BaseModel, Field, model_validator


class Referencia(BaseModel):
    fuente: str = Field(description="Nombre del archivo fuente, ej. '03_mora_y_cobranzas.md'")
    fragmento: str = Field(description="Cita breve (una oración) del texto usado para responder")


class RAGResponse(BaseModel):
    respuesta: str = Field(description="Respuesta a la pregunta, o exactamente 'No lo sé'")
    encontrado: bool = Field(description="True si la respuesta está en el contexto, False si no")
    referencias: list[Referencia] = Field(
        default_factory=list,
        description="Referencias que respaldan la respuesta; vacío si encontrado=False",
    )

    @model_validator(mode="after")
    def referencias_consistentes(self) -> "RAGResponse":
        if not self.encontrado and self.referencias:
            raise ValueError("Si encontrado=False, 'referencias' debe estar vacía")
        if self.encontrado and not self.referencias:
            raise ValueError("Si encontrado=True, 'referencias' debe tener al menos un elemento")
        return self
