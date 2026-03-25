"""Tests for manifest parsing."""

from pathlib import Path

import pytest

from depwatch.inference_service.services.manifest_parser import (
    ManifestParseError,
    detect_ecosystem,
    parse_go_mod,
    parse_manifest,
    parse_package_json,
    parse_requirements_txt,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_manifests"


class TestDetectEcosystem:
    def test_requirements_txt(self) -> None:
        assert detect_ecosystem("requirements.txt") == "pypi"

    def test_package_json(self) -> None:
        assert detect_ecosystem("package.json") == "npm"

    def test_go_mod(self) -> None:
        assert detect_ecosystem("go.mod") == "go"

    def test_unsupported(self) -> None:
        with pytest.raises(ManifestParseError, match="Unsupported"):
            detect_ecosystem("Gemfile")

    def test_with_path_prefix(self) -> None:
        assert detect_ecosystem("/some/path/requirements.txt") == "pypi"


class TestParseRequirementsTxt:
    def test_basic(self) -> None:
        content = "flask>=2.0\nrequests==2.31.0\n"
        deps = parse_requirements_txt(content)
        assert len(deps) == 2
        assert deps[0].name == "flask"
        assert deps[0].ecosystem == "pypi"
        assert deps[1].name == "requests"

    def test_comments_and_blanks(self) -> None:
        content = "# comment\n\nflask\n  # another\n"
        deps = parse_requirements_txt(content)
        assert len(deps) == 1
        assert deps[0].name == "flask"

    def test_extras(self) -> None:
        content = "sqlalchemy[asyncio]>=2.0\n"
        deps = parse_requirements_txt(content)
        assert len(deps) == 1
        assert deps[0].name == "sqlalchemy"

    def test_skip_flags(self) -> None:
        content = "-r dev.txt\n-c constraints.txt\nflask\n"
        deps = parse_requirements_txt(content)
        assert len(deps) == 1

    def test_environment_markers(self) -> None:
        content = 'pywin32; sys_platform == "win32"\nflask\n'
        deps = parse_requirements_txt(content)
        assert len(deps) == 2
        assert deps[0].name == "pywin32"

    def test_inline_comment(self) -> None:
        content = "flask>=2.0  # web framework\n"
        deps = parse_requirements_txt(content)
        assert len(deps) == 1
        assert deps[0].name == "flask"

    def test_normalizes_name(self) -> None:
        content = "My_Package.Name\n"
        deps = parse_requirements_txt(content)
        assert deps[0].name == "my-package-name"

    def test_fixture_file(self) -> None:
        content = (FIXTURES / "requirements.txt").read_text()
        deps = parse_requirements_txt(content)
        names = [d.name for d in deps]
        assert "flask" in names
        assert "requests" in names
        assert "numpy" in names
        assert "sqlalchemy" in names
        assert "gunicorn" in names
        assert len(deps) == 5  # -r line is skipped


class TestParsePackageJson:
    def test_basic(self) -> None:
        content = '{"dependencies": {"express": "^4.18.0"}, "devDependencies": {"jest": "^29.0.0"}}'
        deps = parse_package_json(content)
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert names == {"express", "jest"}
        assert all(d.ecosystem == "npm" for d in deps)

    def test_no_deps(self) -> None:
        content = '{"name": "test"}'
        deps = parse_package_json(content)
        assert deps == []

    def test_invalid_json(self) -> None:
        with pytest.raises(ManifestParseError, match="Invalid"):
            parse_package_json("{bad json")

    def test_dedup_across_sections(self) -> None:
        content = '{"dependencies": {"a": "1"}, "devDependencies": {"a": "2"}}'
        deps = parse_package_json(content)
        assert len(deps) == 1

    def test_fixture_file(self) -> None:
        content = (FIXTURES / "package.json").read_text()
        deps = parse_package_json(content)
        names = {d.name for d in deps}
        assert names == {"express", "lodash", "jest"}


class TestParseGoMod:
    def test_require_block(self) -> None:
        content = "module example.com/app\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n"
        deps = parse_go_mod(content)
        assert len(deps) == 1
        assert deps[0].name == "github.com/gin-gonic/gin"
        assert deps[0].ecosystem == "go"

    def test_single_require(self) -> None:
        content = "module example.com/app\n\nrequire github.com/pkg/errors v0.9.1\n"
        deps = parse_go_mod(content)
        assert len(deps) == 1
        assert deps[0].name == "github.com/pkg/errors"

    def test_inline_comments(self) -> None:
        content = "require (\n\tgithub.com/a/b v1.0 // indirect\n)\n"
        deps = parse_go_mod(content)
        assert len(deps) == 1
        assert deps[0].name == "github.com/a/b"

    def test_fixture_file(self) -> None:
        content = (FIXTURES / "go.mod").read_text()
        deps = parse_go_mod(content)
        names = [d.name for d in deps]
        assert "github.com/gin-gonic/gin" in names
        assert "github.com/stretchr/testify" in names
        assert "github.com/go-sql-driver/mysql" in names
        assert len(deps) == 3


class TestParseManifest:
    def test_routes_to_correct_parser(self) -> None:
        deps = parse_manifest("requirements.txt", "flask\n")
        assert deps[0].ecosystem == "pypi"

        deps = parse_manifest("package.json", '{"dependencies": {"a": "1"}}')
        assert deps[0].ecosystem == "npm"

        deps = parse_manifest("go.mod", "module x\n\nrequire github.com/a/b v1\n")
        assert deps[0].ecosystem == "go"
