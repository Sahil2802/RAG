# Shared engine state, populated once at startup (see app.py lifespan).
# Lives in its own module so both app.py and the routes can import it
# without a circular import.
engine: dict = {}
