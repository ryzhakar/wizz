from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func
# SQLAlchemy does not map BigInt to Int by default on the sqlite dialect.
# It should, but it doesnt.

BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(postgresql.BIGINT(), 'postgresql')
BigIntegerType = BigIntegerType.with_variant(mysql.BIGINT(), 'mysql')
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), 'sqlite')


class Base(AsyncAttrs, DeclarativeBase):
    """Base model class with ids and timestamps."""

    @declared_attr
    def __tablename__(cls) -> str:  # noqa: N805
        """Use the class name as the table name."""
        return cls.__name__.lower()  # type: ignore

    id = Column(
        BigIntegerType,
        primary_key=True,
        autoincrement=True,
    )
    created = Column(
        DateTime(timezone=True),
        default=func.now(),
        index=True,
    )
    updated = Column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )
