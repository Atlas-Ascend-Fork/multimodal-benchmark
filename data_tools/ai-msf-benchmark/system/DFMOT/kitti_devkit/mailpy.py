# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

LOGGER = logging.getLogger(__name__)


class Mail:
    """Dummy class to print messages without sending e-mails"""

    def __init__(self, mailaddress):
        pass

    @staticmethod
    def msg(msg):
        LOGGER.info("%s", msg)

    @staticmethod
    def finalize(success, benchmark, sha_key, mailaddress=None):
        if success:
            LOGGER.info(
                "%s",
                "Results for %s (system: %s) sucessfully created"
                % (benchmark, sha_key),
            )
        else:
            LOGGER.info(
                "%s",
                "Creating results for %s (system: %s) failed" % (benchmark, sha_key),
            )
