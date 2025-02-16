# This file is part of SemanticFrontEndFilter.
#
# SemanticFrontEndFilter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SemanticFrontEndFilter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SemanticFrontEndFilter.  If not, see <https://www.gnu.org/licenses/>.


from setuptools import find_packages
from distutils.core import setup

setup(
    name="semantic_front_end_filter",
    version="0.1.1",
    author="Anqiao Li, Chenyu Yang",
    author_email="anqiali@student.ethz.ch, chenyang@student.ethz.ch",
    packages=find_packages(),
    python_requires=">=3.8",
    description="The package for semantic front end filter",
    install_requires=[""],
)