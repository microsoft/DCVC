# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..models.waseda import (
    Cheng2020Anchor
)

from ..models.priors import (
    FactorizedPrior,
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors
)

model_architectures = {
    "bmshj2018-factorized": FactorizedPrior,
    "bmshj2018-hyperprior": ScaleHyperprior,
    "mbt2018-mean": MeanScaleHyperprior,
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "cheng2020-anchor": Cheng2020Anchor,
}
