"""Historical image validation provenance, indexed by exact algorithm and phase.

These are preserved facts, including reports unavailable in a source checkout.
Do not infer evidence from newly registered algorithms, bindings or phases.
Ordinary package data in Python keeps installed use independent of outputs/.
"""

from mllm_comp.core.support import EvidenceKind, EvidenceRecord
from mllm_comp.core.vocabulary import ExecutionPhase


# Run -> (shared provenance, backbone -> (report, report checksum, physical GPU)).
_VALIDATION_RUNS = {
    "3x3_forward_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_3x3_forward_validation_20260811.json",
            "checksum": "9637aa045e97ccd2391cd9d6a4dd9c05ce718af3c42d11b0363baaae0d850985",
            "source_commit": "64e49105135bcb0d9859e2da1f7ca28356abc82c",
            "recorded_at": "2026-08-11T09:30:00+08:00",
        },
        {
            "llava_onevision": (
                "outputs/llava_onevision_image_compression_forward_20260811.json",
                "1e5e54f247f959a7497f1a03654164410629b0cddb7b84e7b2a4cc45fc31dd33",
                1,
            ),
            "qwen25_vl": (
                "outputs/qwen25_vl_image_compression_forward_20260811.json",
                "59685c636608d18d84854473582e10a5794a579f9fe975eba4ef48c464946851",
                1,
            ),
            "qwen3_vl": (
                "outputs/qwen3_vl_image_compression_forward_20260811.json",
                "29ff87d8ffb8e45f81a21edcad1803da0cfef21cc55517f0e403c770e01cca25",
                1,
            ),
        },
    ),
    "qwen_supplied_saint_aim_pact_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_qwen_supplied_saint_aim_pact_validation_20260819.json",
            "checksum": "c015fbd4ff8314877110066828c8a758d36eebac470b547253ca6f4acbd64413",
            "source_commit": "77c23eb9fcd4d01b64b80ef83e6622daf9eecac1",
            "recorded_at": "2026-08-19T02:55:00+08:00",
        },
        {
            "qwen25_vl": (
                "outputs/remote-qwen-supplied-candidate-20260819/qwen25_vl_forward.json",
                "f63f9d4a1b3b25acb52fdb5e615ee247a2d2942d534f106cedea50a23727bb6b",
                0,
            ),
            "llava_onevision": (
                "outputs/remote-qwen-supplied-candidate-20260819/llava_onevision_forward.json",
                "3f67901adbb2807ec5a766248af2845519580f56ea061b0c5cf9a48c14baa882",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-qwen-supplied-candidate-20260819/qwen3_vl_forward.json",
                "5fabbc5c547167cb99c271e03a3ae613c49bb18da067a38e61cf75cb33e7d608",
                0,
            ),
        },
    ),
    "visionzip_forward_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_visionzip_forward_validation_20260818.json",
            "checksum": "8552ee79f7d8c0a9ee34a0e47b0a5fd7eb5298d1819f8fe065337bb26f010baa",
            "source_commit": "ec072bcdb8ce9294d1a21eecd9c42485852d171b",
            "recorded_at": "2026-08-18T12:12:51+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-visionzip-candidate-ec072bc/llava_onevision_forward.json",
                "a28e2d748e9fc1d45db0f096bd14ea99624fe975a5bc0971f9caf35e81ff373f",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-visionzip-candidate-ec072bc/qwen25_vl_forward.json",
                "b5c8bcf82a27925233a8920797db1c8e506505e0fa4f64c6c9d7a81d3ebb46bb",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-visionzip-candidate-ec072bc/qwen3_vl_forward.json",
                "296b9caabd6b53ec6ecaa25998f0191b19b23b6cfe1ce707fa50c0938a1444b6",
                0,
            ),
        },
    ),
    "divprune_forward_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_divprune_forward_validation_20260817.json",
            "checksum": "e089b02f3b49bee09419a38eb65b3e0833282ec594a6755f91ffe2d38c4dac56",
            "source_commit": "abdd7c0ec47a46ec6872f46f1a73ed919aca2b5f",
            "recorded_at": "2026-08-17T19:16:50+08:00",
        },
        {
            "llava_onevision": (
                "outputs/divprune-smoke/llava_onevision_compression_candidate_abdd7c0.json",
                "0e1718d7399bb5c19bf24a9acb763607bf05d55a3003cca7488423017e9e15ec",
                1,
            ),
            "qwen25_vl": (
                "outputs/divprune-smoke/qwen25_vl_compression_candidate_abdd7c0.json",
                "30990d9901cbfb79562a49bf832eb0165e6238bdba3d71f10230927656950209",
                1,
            ),
            "qwen3_vl": (
                "outputs/divprune-smoke/qwen3_vl_compression_candidate_abdd7c0.json",
                "a463f5db55a5dc156d79725ed1f6f9b785ef48c7bc0d4cb6a5b393d7a465dc4a",
                1,
            ),
        },
    ),
    "mmtok_forward_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_mmtok_forward_validation_20260818.json",
            "checksum": "c9b62ee65f1131098f95e44e83ed75c25b6584316f6ef753b260d07fe79df837",
            "source_commit": "05c93dbefbd37c8b4a77023f6c9d37534b933fad",
            "recorded_at": "2026-08-18T04:32:45+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-mmtok-candidate-05c93db/llava_onevision_mmtok_forward.json",
                "fd20549733a70544dd42990cf5c0a97fa7c46733ef9c428d08985d0b38b00010",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-mmtok-candidate-05c93db/qwen25_vl_mmtok_forward.json",
                "d7629a120bc8b4a65e9b29d249077ecbe528aed6f668ee1196d2ee92cde04db4",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-mmtok-candidate-05c93db/qwen3_vl_mmtok_forward.json",
                "e1f6a02b8286a30e1ba7bb24d776a827f5f64e8a52c0ec16a42aeca25b0a5673",
                1,
            ),
        },
    ),
    "vispruner_forward_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_vispruner_forward_validation_20260818.json",
            "checksum": "f6de5eb8e0b735953440de02d818e47171ffd421e8b2353e4e332fc88d707dd4",
            "source_commit": "274f2f5d9732a948e8fc01f75b7186e424eddba6",
            "recorded_at": "2026-08-18T05:15:38+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-vispruner-candidate-274f2f5/llava_onevision_vispruner_forward.json",
                "c9eec29f6f01429f49700ed54b9d0af2d995a527b4d1ceb1df9787a716a824ec",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-vispruner-candidate-274f2f5/qwen25_vl_vispruner_forward.json",
                "a0d588f7c62c8be7bc1f8bc44d863dc27b8f4c008bce9c61f8216309d959f62b",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-vispruner-candidate-274f2f5/qwen3_vl_vispruner_forward.json",
                "5958fea86db32a9efc3916f95eb2d72ff85e1ad46e4a33196adbcc31503b7b4b",
                1,
            ),
        },
    ),
    "zoo_prune_forward_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_zoo_prune_forward_validation_20260818.json",
            "checksum": "58d7680c76e193f1cdcb1abc8525ff1f1cf8ed88ada06c27f53e684633d0d14e",
            "source_commit": "b030ee881411a8740015f7195b4c3abfcdd69f15",
            "recorded_at": "2026-08-18T06:05:08+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-zoo-prune-candidate-b030ee8/llava_onevision_zoo_prune_forward.json",
                "c62ff4bf16de1cc63c779c4174c12397958f1f3a27ecefceb878e60194b887b9",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-zoo-prune-candidate-b030ee8/qwen25_vl_zoo_prune_forward.json",
                "2adee850c2821ad2468cc175038a800e37ed17e9f771ed765ecefd1c3c6c6e38",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-zoo-prune-candidate-b030ee8/qwen3_vl_zoo_prune_forward.json",
                "b75445d31134d396ce66ca9cf2adc2d1a2bd9d0798b5a231f8ed5ae7f3837644",
                1,
            ),
        },
    ),
    "prumerge_gap_feather_ivc_validation:forward": (
        {
            "artifact": "outputs/imagellm/imagellm_prumerge_gap_feather_ivc_validation_20260819.json",
            "checksum": "14ccfdec1fe7cdd610dd7e125d5f90f07c790e5e7ec88466de05c83bc20426f5",
            "source_commit": "6f1f448727a9c07359e675d8f82367f369150a48",
            "recorded_at": "2026-08-19T03:55:00+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-prumerge-gap-feather-ivc-20260819/llava_onevision_forward.json",
                "12b780b5a33b2e8173a11a9be0d040aaee464fd15b2acb584f7f985557dfadee",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-prumerge-gap-feather-ivc-20260819/qwen25_vl_forward.json",
                "473df1ab488806e42d8b1875f12ccadc684f56113c9af0bb7d34dfb2e61c33e9",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-prumerge-gap-feather-ivc-20260819/qwen3_vl_forward.json",
                "3e4f5d1340ba4a055dd5c9476864050eac3cace0c141ebf17b619b88fdde212e",
                0,
            ),
        },
    ),
    "qwen_supplied_saint_aim_pact_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_qwen_supplied_saint_aim_pact_validation_20260819.json",
            "checksum": "c015fbd4ff8314877110066828c8a758d36eebac470b547253ca6f4acbd64413",
            "source_commit": "77c23eb9fcd4d01b64b80ef83e6622daf9eecac1",
            "recorded_at": "2026-08-19T02:55:00+08:00",
        },
        {
            "qwen25_vl": (
                "outputs/remote-qwen-supplied-candidate-20260819/qwen25_vl_pact_generate.json",
                "ca3f669ecba314707a650a734144a64a4ab33f82a350cfff03d84e0261955301",
                0,
            ),
            "llava_onevision": (
                "outputs/remote-qwen-supplied-candidate-20260819/llava_onevision_pact_generate.json",
                "76adb561de3d481f8e14e172e5941c36345d1a008cfa3996fc076402133df8f9",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-qwen-supplied-candidate-20260819/qwen3_vl_pact_generate.json",
                "11e762fc56d48163181873f18f85d036faebd9aa2234a97896f8d32d655be618",
                0,
            ),
        },
    ),
    "pyramid_fastv_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_pyramid_fastv_generate_validation_20260818.json",
            "checksum": "359c20a71ccd4b57f22414232edaab42f7d79f4de29b7b92ada8e4872892d561",
            "source_commit": "63bdd324631bdddb2547647218e1fd8ac43c4764",
            "recorded_at": "2026-08-18T12:20:35+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-pyramid-fastv-candidate-63bdd32/llava_onevision_decoder_generate.json",
                "e179380f1d2d86c46c625e311877f89f76f660c7cb4ee08b1b7d5252ad241702",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-pyramid-fastv-candidate-63bdd32/qwen25_vl_decoder_generate.json",
                "4bad55717a5992ce73fb1c957128b18af66bdd484936fa1fcc2a22bf4ba76b6f",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-pyramid-fastv-candidate-63bdd32/qwen3_vl_decoder_generate.json",
                "e610be22131e0ac1a5fe5e68b55e7bacadd09c4ddf987ea459b313bab9878376",
                0,
            ),
        },
    ),
    "sgl_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_sgl_generate_validation_20260818.json",
            "checksum": "8e95e949e75b38bcce6ae51b72c5372728d874e115fc2b18ec94547badb9f5c3",
            "source_commit": "107bbd0051f15829ae8c79342ec5bc4ecfd31e7e",
            "recorded_at": "2026-08-18T10:44:18+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-sgl-candidate-107bbd0/llava_onevision_decoder_generate.json",
                "5f46d263418901968d345d4bf7195c13926f4898941b4fc3be64f89c8ae1749a",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-sgl-candidate-107bbd0/qwen25_vl_decoder_generate.json",
                "bd7436fcb0b3281926f163efa919e291ee85cab05dfac7ef4365fcf039a6bf61",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-sgl-candidate-107bbd0/qwen3_vl_decoder_generate.json",
                "72442def5da4db4e4bedc21770cb86fd41f26410e077f0c7ab99453807d05417",
                0,
            ),
        },
    ),
    "dart_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_dart_generate_validation_20260817.json",
            "checksum": "69c0604c49c358683c277af06659b02436e4c8953d4f9324d3ef5115dd7bd80f",
            "source_commit": "3d9cb05e63e8d5c743628a2563c94f03a3826df7",
            "recorded_at": "2026-08-17T22:25:00+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-g5-candidate-3d9cb05e-dart/llava_onevision_decoder_generate.json",
                "8c408bcbc3a2a57f1685914eea42cea906dd743070b8b1ba9384040f3c2cba2e",
                1,
            ),
            "qwen25_vl": (
                "outputs/remote-g5-candidate-3d9cb05e-dart/qwen25_vl_decoder_generate.json",
                "2ca72cf510da0364b661b62c5031c6b3acd170a3951fae22557a05884fe249df",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-g5-candidate-3d9cb05e-dart/qwen3_vl_decoder_generate.json",
                "9ca7cea7fdf96b72c46234c31ffb1ca518cf31ff62b23265a5ed31f38dccf8a5",
                1,
            ),
        },
    ),
    "sparsevlm_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_sparsevlm_generate_validation_20260817.json",
            "checksum": "9286dad8c6ce60fc12064ec36efef2b319715c29cdc1f4936f06bfc91d60314e",
            "source_commit": "9c96ec416212a20a7d986bc4882e5ccd13271bd7",
            "recorded_at": "2026-08-17T23:39:22+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-sparsevlm-candidate-9c96ec4/llava_onevision_decoder_generate.json",
                "e2287e51d09a60fb39e0a3588e5f896adad291ce8f24963b795c5ef268bed9b7",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-sparsevlm-candidate-9c96ec4/qwen25_vl_decoder_generate.json",
                "c265d97fd370ca4e19f9c9087d7bd977fbe8fc5ee223ffb11b39822943e325c6",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-sparsevlm-candidate-9c96ec4/qwen3_vl_decoder_generate.json",
                "14f4e23017370a23008396be1a663847e562a919652f48be46ab9afabe02ad9c",
                0,
            ),
        },
    ),
    "finepruner_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_finepruner_generate_validation_20260818.json",
            "checksum": "6b3f81d6cee8b926426f45c17ed5bcd1941265eb8a0dd0306213237aaca24ccd",
            "source_commit": "e82c069fd77f0ded3f066cef70612fb791b2d0be",
            "recorded_at": "2026-08-18T01:08:21+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-finepruner-candidate-e82c069/llava_onevision_decoder_generate.json",
                "8720a1350f35886ba0a84d4e5c04fe800f574331f063849dd8e232a5e4b823ee",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-finepruner-candidate-e82c069/qwen25_vl_decoder_generate.json",
                "9d0ec9d5d7be5aa4929a74a32cd2e5b9234a6cc8aa9e1e30c60f93b9726bbca5",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-finepruner-candidate-e82c069/qwen3_vl_decoder_generate.json",
                "b8cfd9c044fff2859132e88ee951384badf6daa6740f3c75481b4bd314ad4561",
                1,
            ),
        },
    ),
    "btp_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_btp_generate_validation_20260818.json",
            "checksum": "c6c0c26dab327f3b559299553a79db43bad5f3b177a29114c3ce4da497023473",
            "source_commit": "db648cdc1767e805012753b3c3b287089cf86b07",
            "recorded_at": "2026-08-18T02:03:20+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-btp-candidate-db648cd/llava_onevision_decoder_generate.json",
                "d59cf606053844098552e780d84e16f161a6c48d161ca99a60dc6d82af3eaf24",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-btp-candidate-db648cd/qwen25_vl_decoder_generate.json",
                "8e8758318bf064bf1da4cd9e4e75deb70a5347df8c01f1baa4eb413e35750150",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-btp-candidate-db648cd/qwen3_vl_decoder_generate.json",
                "f5c887481a6dea7e1a8bcc026310feb1e0d0425c5673c517263d51bb71938fe8",
                0,
            ),
        },
    ),
    "halfv_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_halfv_generate_validation_20260818.json",
            "checksum": "1372298c0aeda6676a6499dd8c3fbd430d768e3cba513e284c6176d2674e5a8f",
            "source_commit": "bf42455ebe11b7c6901e42fe38ea34c159a40acf",
            "recorded_at": "2026-08-18T02:49:56+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-halfv-candidate-bf42455/llava_onevision_decoder_generate.json",
                "01ded521c8cbaff61a347d123d52a7cca711e6ec9a758f4da0ea814dfa312de6",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-halfv-candidate-bf42455/qwen25_vl_decoder_generate.json",
                "7389f0cc87a895feef761edcb4ba67cb15daac7cd967e09e9a3e3ac56f37eb3e",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-halfv-candidate-bf42455/qwen3_vl_decoder_generate.json",
                "32338925aea1a8455e44545aac74b4177f23675600fdc4e09b7f40741c763108",
                0,
            ),
        },
    ),
    "transprune_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_transprune_generate_validation_20260818.json",
            "checksum": "633baa2efe676cdeed41ec8bc147835bf97a1a35d9c0155b55b41860b6d56c40",
            "source_commit": "e0b8297e160edbfac7a2587490b7ec774d8dfb8b",
            "recorded_at": "2026-08-18T03:34:00+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-transprune-candidate-e0b8297/llava_onevision_decoder_generate.json",
                "f528447d574ca15336a147cb35ee0f7e0d28933ef4a960e3ac0030b3ad711851",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-transprune-candidate-e0b8297/qwen25_vl_decoder_generate.json",
                "6b9f80d7dade88007ca36513196abe9fbc3219ab415e9e734070385b0788c58d",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-transprune-candidate-e0b8297/qwen3_vl_decoder_generate.json",
                "05f4674f50b206035313e3ee2cca88d3319ccae6dec0d04b431eed31ba97fcfb",
                0,
            ),
        },
    ),
    "nuwa_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_nuwa_generate_validation_20260818.json",
            "checksum": "a1f6dcb687f293ff1945fd1af40caf5d75c221f10e1768071129f7946f878293",
            "source_commit": "bb4373552daa47cce04835f5940227273ee39d89",
            "recorded_at": "2026-08-18T07:20:36+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-nuwa-candidate-bb43735/llava_onevision_decoder_generate.json",
                "f8166d1cdcc3fc93597ec508b03788bd00fa672afaec229a67e6e8b98de840f0",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-nuwa-candidate-bb43735/qwen25_vl_decoder_generate.json",
                "d1d3af580b273c8464e815ae115780468d64b2558b3d2ba1cf22e4b0d1110549",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-nuwa-candidate-bb43735/qwen3_vl_decoder_generate.json",
                "0d4eafb1d69fc557b05d8faaa37ec3097ee3fb72b8c3dcfe2375ac4c955ebd21",
                1,
            ),
        },
    ),
    "restore_generate_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_restore_generate_validation_20260818.json",
            "checksum": "93ad518591b21ad7a5fdb7133675c76208c40796b988107651dbd1321b01a048",
            "source_commit": "3604a651512da886221f4e68bd418e4f1e26bc84",
            "recorded_at": "2026-08-18T08:56:50+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-restore-candidate-c67c8dc/llava_onevision_decoder_generate.json",
                "c887a50c89e73d324acdeed46031948ff881ce996c2ed9528f32268e98eef94e",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-restore-candidate-c67c8dc/qwen25_vl_decoder_generate.json",
                "d632b0e8024b675190bc6d5702bf30ac3e296c28e1ec45ce01dc2701a5717e4f",
                0,
            ),
            "qwen3_vl": (
                "outputs/remote-restore-candidate-c67c8dc/qwen3_vl_decoder_generate.json",
                "aca443d900d8c88cc397a6334cc3763c5f396f824968fe6dc497f0a3af1afddb",
                1,
            ),
        },
    ),
    "prumerge_gap_feather_ivc_validation:generate": (
        {
            "artifact": "outputs/imagellm/imagellm_prumerge_gap_feather_ivc_validation_20260819.json",
            "checksum": "14ccfdec1fe7cdd610dd7e125d5f90f07c790e5e7ec88466de05c83bc20426f5",
            "source_commit": "6f1f448727a9c07359e675d8f82367f369150a48",
            "recorded_at": "2026-08-19T03:55:00+08:00",
        },
        {
            "llava_onevision": (
                "outputs/remote-prumerge-gap-feather-ivc-20260819/llava_onevision_decoder_generate.json",
                "16c40a96c0263bd3358d3e91bd8c966f51a81c6ff4bfcaa4ef9bc9aae23823aa",
                0,
            ),
            "qwen25_vl": (
                "outputs/remote-prumerge-gap-feather-ivc-20260819/qwen25_vl_decoder_generate.json",
                "17562e73d86df8f189bbf88eaae5a657fad00777988463371d1f0012e82535a4",
                1,
            ),
            "qwen3_vl": (
                "outputs/remote-prumerge-gap-feather-ivc-20260819/qwen3_vl_decoder_generate.json",
                "a0c66fab8e3ed14e24121d7bb5a53e916d0f32a35ea1c9d3258f08ec98f17846",
                0,
            ),
        },
    ),
}

# Only the exact algorithm/phase pairs below have validation evidence.
_CELL_RUNS = {
    ("tome", "forward"): "3x3_forward_validation:forward",
    ("saint", "forward"): "qwen_supplied_saint_aim_pact_validation:forward",
    ("aim", "forward"): "qwen_supplied_saint_aim_pact_validation:forward",
    ("pact", "forward"): "qwen_supplied_saint_aim_pact_validation:forward",
    ("visionzip", "forward"): "visionzip_forward_validation:forward",
    ("pyramid_drop", "forward"): "3x3_forward_validation:forward",
    ("divprune", "forward"): "divprune_forward_validation:forward",
    ("mmtok", "forward"): "mmtok_forward_validation:forward",
    ("vispruner", "forward"): "vispruner_forward_validation:forward",
    ("zoo_prune", "forward"): "zoo_prune_forward_validation:forward",
    ("prumerge", "forward"): "prumerge_gap_feather_ivc_validation:forward",
    ("gap", "forward"): "prumerge_gap_feather_ivc_validation:forward",
    ("pact", "generate"): "qwen_supplied_saint_aim_pact_validation:generate",
    ("pyramid_drop", "generate"): "pyramid_fastv_generate_validation:generate",
    ("fastv", "generate"): "pyramid_fastv_generate_validation:generate",
    ("sgl", "generate"): "sgl_generate_validation:generate",
    ("dart", "generate"): "dart_generate_validation:generate",
    ("sparsevlm", "generate"): "sparsevlm_generate_validation:generate",
    ("finepruner", "generate"): "finepruner_generate_validation:generate",
    ("btp", "generate"): "btp_generate_validation:generate",
    ("halfv", "generate"): "halfv_generate_validation:generate",
    ("transprune", "generate"): "transprune_generate_validation:generate",
    ("nuwa", "generate"): "nuwa_generate_validation:generate",
    ("restore", "generate"): "restore_generate_validation:generate",
    ("feather", "generate"): "prumerge_gap_feather_ivc_validation:generate",
    ("ivc_prune", "generate"): "prumerge_gap_feather_ivc_validation:generate",
}

_TRAIN_STEP_PHASES = {
    "tome": "forward",
    "saint": "forward",
    "aim": "forward",
    "pact": "forward",
    "visionzip": "forward",
    "pyramid_drop": "forward",
    "divprune": "forward",
    "mmtok": "forward",
    "vispruner": "forward",
    "zoo_prune": "forward",
    "prumerge": "forward",
    "gap": "forward",
    "fastv": "generate",
    "sgl": "generate",
    "dart": "generate",
    "sparsevlm": "generate",
    "finepruner": "generate",
    "btp": "generate",
    "halfv": "generate",
    "transprune": "generate",
    "nuwa": "generate",
    "restore": "generate",
    "feather": "generate",
    "ivc_prune": "generate",
}

_TRAIN_STEP_REPORTS = {
    "llava_onevision": (
        "outputs/imagellm/train_step/llava_onevision_train_step_20260819.json",
        "63ba0139fdbbf7fbfc1189660d7958b2b32063c7f53e6a73717ba87396e87dd3",
        0,
    ),
    "qwen25_vl": (
        "outputs/imagellm/train_step/qwen25_vl_train_step_20260819.json",
        "2ffb8c6fd86c08beac1e76153379f82601618fd31f27cf897bbd3f23a6f219be",
        0,
    ),
    "qwen3_vl": (
        "outputs/imagellm/train_step/qwen3_vl_train_step_20260819.json",
        "a1ca8eb83b100d57b247829630328c0ef0ba6998d25b66c3c58323f6ce352cfe",
        1,
    ),
}

_TRAIN_STEP_COMMON = {
    "source_commit": "2c2e575ee83b66260ca72b50ecfe3070254c76f6",
    "recorded_at": "2026-08-19T05:45:00+08:00",
}

_TRAIN_STEP_METADATA = {
    "dataset": "/home/WuHao/datasets/EITNLP-Img-Post-dataset/train_10k.jsonl",
    "dataset_record_id": "VG_100K/2334701_8",
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "optimizer_steps": 1,
    "teacher_forcing": True,
    "use_cache": False,
    "real_compression_verified": True,
    "finite_loss_verified": True,
    "finite_nonzero_gradient_verified": True,
    "optimizer_step_verified": True,
    "layout_repair_verified": True,
    "gradient_checkpoint_replay_verified": True,
}


def image_evidence(
    *,
    algorithm: str,
    backbone: str,
    binding: str,
    phase: ExecutionPhase,
) -> tuple[EvidenceRecord, ...]:
    """Build records from explicitly recorded validation and training cells."""
    common, reports = _VALIDATION_RUNS[_CELL_RUNS[(algorithm, phase.value)]]
    report, report_sha256, physical_gpu = reports[backbone]
    generate = phase is ExecutionPhase.GENERATE
    records = (
        EvidenceRecord(
            kind=EvidenceKind.KERNEL,
            passed=True,
            **common,
            metadata={"algorithm": algorithm, "scope": "backbone_neutral_image_kernel"},
        ),
        EvidenceRecord(
            kind=EvidenceKind.CPU_CONTRACT,
            passed=True,
            **common,
            metadata={
                "binding": binding,
                "scope": (
                    "provider_prefill_decode_cache_layout_contracts"
                    if generate
                    else "provider_adapter_layout_runtime_contracts"
                ),
            },
        ),
        EvidenceRecord(
            kind=EvidenceKind.REAL_CHECKPOINT,
            passed=True,
            **common,
            metadata={
                "binding": binding,
                "report": report,
                "report_sha256": report_sha256,
                "physical_gpu": physical_gpu,
                "use_cache": generate,
                "provider_auto_installed": True,
                **(
                    {"max_new_tokens": 2, "cache_telemetry_verified": True}
                    if generate
                    else {"cache_telemetry_applicable": False}
                ),
            },
        ),
    )
    if _TRAIN_STEP_PHASES.get(algorithm) == phase.value:
        artifact, checksum, physical_gpu = _TRAIN_STEP_REPORTS[backbone]
        records += (
            EvidenceRecord(
                kind=EvidenceKind.TRAIN_STEP,
                passed=True,
                artifact=artifact,
                checksum=checksum,
                **_TRAIN_STEP_COMMON,
                metadata={
                    **_TRAIN_STEP_METADATA,
                    "algorithm": algorithm,
                    "binding": binding,
                    "backbone": backbone,
                    "execution_phase": phase.value,
                    "physical_gpu": physical_gpu,
                },
            ),
        )
    return records
