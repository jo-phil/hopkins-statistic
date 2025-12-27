.PHONY: docs clean

docs:
	uv run pygmentize -S github-light-default -f html -a .pdoc-code \
	    > docs/templates/syntax-highlighting.css
	echo "@media (prefers-color-scheme: dark) {" \
	    >> docs/templates/syntax-highlighting.css
	uv run pygmentize -S github-dark-default -f html -a .pdoc-code \
	    >> docs/templates/syntax-highlighting.css
	echo "}" >> docs/templates/syntax-highlighting.css

	uv run pdoc --docformat google \
		--math --no-show-source --no-search \
		--template-directory docs/templates \
		-o docs/html src/hopkins_statistic

	mv -f docs/html/hopkins_statistic.html docs/html/index.html


clean:
	rm -rf docs/html/
	rm -f docs/templates/syntax-highlighting.css
