# Development

```{include} ../../README.md
:start-after: "## Development"
:end-before: "## License"
```

Build these docs locally:

```
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

## License

```{include} ../../README.md
:start-after: "## License"
```
