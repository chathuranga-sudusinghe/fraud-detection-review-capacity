# Analysis run 2026-10-29T013414Z

Generated from the JSON records of this run directory; not edited by hand.

| Field | Value |
|---|---|
| `run_type` | `"analysis"` |
| `run_id` | `"2026-10-29T013414Z"` |
| `started` | `"2026-10-29T01:34:17.893690Z"` |
| `outcome` | `"completed"` |
| `finished` | `"2026-10-29T01:35:40.550960Z"` |
| `error` | — |
| `records` | `23` |

## Run identity

| Field | Value |
|---|---|
| `identity.run_id` | `"2026-10-29T013414Z"` |
| `identity.run_type` | `"analysis"` |
| `identity.code_commit` | `"6e2ff564e7b5a0dc505b7ee9c9465b4449da0092"` |
| `identity.uncommitted_changes` | `false` |
| `identity.source_sha256` | `"06540fd34ff853bfdc88ca3bc4ac291f656f392683bb6261f513353667abb5be"` |
| `identity.settings_sha256` | `"84e8b74a49c1d1fc7f56b03df352183da8d7555c4975b8eef6ef5bb67e282d1c"` |
| `identity.declaration_sha256` | `"3b206368e3930880fd6a45b13b2c6a21a52cb4b87b8242813191648b9e0ce4f7"` |
| `identity.dataset_manifest_sha256` | `"eb67c7bdca3509c11abba36147a3ed66406db4caa1570035dc62eeb7910bdfeb"` |
| `identity.seeds.main` | `42` |
| `identity.seeds.additional` | `[43, 44]` |
| `identity.seeds.sampler` | `42` |
| `identity.seeds.preprocessing` | `0` |

## Comparison tables

### Selected settings

For each model, the unweighted and balanced configurations side by side: the selected trial, its V1 objective and length, and every searched parameter (DR-015 §1).

**`lightgbm`**

| Configuration | Family | Selected trial | Trials tied at the best | V1 objective | Retained length | Stopping | learning_rate | num_leaves | min_data_in_leaf | feature_fraction | bagging_fraction | lambda_l2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `5` | `[5]` | `0.42015895953757226` | `843` | `"early_stopping"` | `0.01729915767044926` | `348` | `143` | `0.9049494329519157` | `0.9256684790605736` | `0.0` |
| `LGB-b` | `tree` | `2` | `[2]` | `0.42052023121387283` | `353` | `"early_stopping"` | `0.22601761106979648` | `448` | `1133` | `0.6546592048070723` | `0.9773280372088982` | `0.0` |

**`catboost`**

| Configuration | Family | Selected trial | Trials tied at the best | V1 objective | Retained length | Stopping | depth | learning_rate | l2_leaf_reg | bagging_temperature | leaf_estimation_iterations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Cat-u` | `tree` | `3` | `[3]` | `0.40173410404624277` | `328` | `"early_stopping"` | `9` | `0.04876132783920775` | `6.839996739537394` | `0.08304025522428526` | `10` |
| `Cat-b` | `tree` | `9` | `[9]` | `0.39378612716763006` | `300` | `"early_stopping"` | `6` | `0.23525450703996198` | `1.233052876299107` | `0.20837400521592242` | `10` |

**`resnet`**

| Configuration | Family | Selected trial | Trials tied at the best | V1 objective | Retained length | Stopping | n_layers | d | d_hidden_factor | hidden_dropout | residual_dropout | d_embedding | lr | weight_decay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `RES-u` | `neural` | `8` | `[8]` | `0.3854768786127168` | `15` | `"maximum"` | `10` | `235` | `3.780540683737454` | `0.18971705150221702` | `0.0` | `84` | `0.0003774249693509198` | `0.0002462038927745929` |
| `RES-b` | `neural` | `1` | `[1]` | `0.36560693641618497` | `13` | `"maximum"` | `2` | `128` | `2.0` | `0.25` | `0.1` | `192` | `0.001` | `0.0` |

**`ft_transformer`**

| Configuration | Family | Selected trial | Trials tied at the best | V1 objective | Retained length | Stopping | n_layers | d_token | attention_dropout | ffn_dropout | lr | weight_decay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FTT-u` | `neural` | `5` | `[5]` | `0.36921965317919075` | `8` | `"early_stopping"` | `6` | `504` | `0.08874303541781964` | `0.2873813672806639` | `0.00017352298832476001` | `7.724237714716737e-06` |
| `FTT-b` | `neural` | `6` | `[6]` | `0.3684971098265896` | `15` | `"maximum"` | `2` | `296` | `0.018426853941648558` | `0.28973578677266254` | `0.0002423058075919228` | `7.953327601896765e-06` |

### Search trials

For each configuration, every search trial with every searched parameter, its V1 objective, retained checkpoint, stopping and runtime (DR-015 §1–§2).

**`LGB-u`**

| Configuration | Trial | Status | learning_rate | num_leaves | min_data_in_leaf | feature_fraction | bagging_fraction | lambda_l2 | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LGB-u trial 1` | `1` | `"completed"` | `0.1` | `31` | `20` | `1.0` | `1.0` | `0.0` | `0.4147398843930636` | `851` | `901` | `"early_stopping"` | `48.0112543030009` |
| `LGB-u trial 2` | `2` | `"completed"` | `0.22601761106979648` | `448` | `1133` | `0.6546592048070723` | `0.9773280372088982` | `0.0` | `0.4161849710982659` | `357` | `407` | `"early_stopping"` | `29.752004214002227` |
| `LGB-u trial 3` | `3` | `"completed"` | `0.1290757432936695` | `700` | `39` | `0.6916710240777042` | `0.7896487730540107` | `0.0` | `0.41510115606936415` | `167` | `217` | `"early_stopping"` | `27.455501927001023` |
| `LGB-u trial 4` | `4` | `"completed"` | `0.23475516904426613` | `154` | `28` | `0.9644871396321761` | `0.5072939515473602` | `1.0383864800791283` | `0.4031791907514451` | `416` | `466` | `"early_stopping"` | `31.39960624000014` |
| `LGB-u trial 5` | `5` | `"completed"` | `0.01729915767044926` | `348` | `143` | `0.9049494329519157` | `0.9256684790605736` | `0.0` | `0.42015895953757226` | `843` | `893` | `"early_stopping"` | `102.96825184100089` |
| `LGB-u trial 6` | `6` | `"completed"` | `0.015824883578574592` | `169` | `561` | `0.8881758881483863` | `0.6563225410597064` | `0.1419087232569387` | `0.4042630057803468` | `635` | `685` | `"early_stopping"` | `52.75343602800058` |
| `LGB-u trial 7` | `7` | `"completed"` | `0.08028819960193419` | `431` | `670` | `0.8342502316769134` | `0.5089408954260624` | `2.3035504187312235` | `0.404985549132948` | `308` | `358` | `"early_stopping"` | `21.984864633999678` |
| `LGB-u trial 8` | `8` | `"completed"` | `0.01218697587169257` | `765` | `1163` | `0.8104764121582517` | `0.5679182667759608` | `3.786140960885953` | `0.39197976878612717` | `647` | `697` | `"early_stopping"` | `42.77845129900015` |
| `LGB-u trial 9` | `9` | `"completed"` | `0.02948061549736706` | `167` | `23` | `0.5389603939382202` | `0.7541846170762744` | `0.283456991321791` | `0.40534682080924855` | `256` | `306` | `"early_stopping"` | `23.428700161999586` |
| `LGB-u trial 10` | `10` | `"completed"` | `0.03261780281441039` | `336` | `1114` | `0.78631223180285` | `0.9756775932427404` | `5.82343521174076` | `0.4096820809248555` | `527` | `577` | `"early_stopping"` | `43.8906726919995` |

**`Cat-u`**

| Configuration | Trial | Status | depth | learning_rate | l2_leaf_reg | bagging_temperature | leaf_estimation_iterations | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Cat-u trial 1` | `1` | `"completed"` | `6` | `0.03234` | `3.0` | `1.0` | `10` | `0.39234104046242774` | `465` | `515` | `"early_stopping"` | `57.37730923399795` |
| `Cat-u trial 2` | `2` | `"completed"` | `6` | `0.011711449557806689` | `3.940125087194552` | `0.10727525115747005` | `5` | `0.3927023121387283` | `823` | `873` | `"early_stopping"` | `89.53221522200329` |
| `Cat-u trial 3` | `3` | `"completed"` | `9` | `0.04876132783920775` | `6.839996739537394` | `0.08304025522428526` | `10` | `0.40173410404624277` | `328` | `378` | `"early_stopping"` | `60.79468055399775` |
| `Cat-u trial 4` | `4` | `"completed"` | `6` | `0.01831704828326315` | `2.9762762356702375` | `0.3571015588198677` | `7` | `0.3898121387283237` | `521` | `571` | `"early_stopping"` | `61.084933212005126` |
| `Cat-u trial 5` | `5` | `"completed"` | `8` | `0.12744790121234295` | `1.021409401170329` | `0.8560817972120977` | `4` | `0.398121387283237` | `163` | `213` | `"early_stopping"` | `33.79549941299774` |
| `Cat-u trial 6` | `6` | `"completed"` | `8` | `0.013710329939504001` | `2.4710925897323555` | `0.6679735767142544` | `9` | `0.39197976878612717` | `513` | `563` | `"early_stopping"` | `77.63960415499605` |
| `Cat-u trial 7` | `7` | `"completed"` | `7` | `0.029669263862569156` | `2.257312144928934` | `0.10236368082671032` | `6` | `0.398121387283237` | `494` | `544` | `"early_stopping"` | `65.55074702599813` |
| `Cat-u trial 8` | `8` | `"completed"` | `9` | `0.1777373665066867` | `6.524576921007237` | `0.5720066363244991` | `7` | `0.40101156069364163` | `267` | `317` | `"early_stopping"` | `51.53361502599728` |
| `Cat-u trial 9` | `9` | `"completed"` | `6` | `0.23525450703996198` | `1.233052876299107` | `0.20837400521592242` | `10` | `0.39414739884393063` | `100` | `150` | `"early_stopping"` | `18.68758815000183` |
| `Cat-u trial 10` | `10` | `"completed"` | `7` | `0.047617053381003464` | `9.490466511636619` | `0.7420573573979781` | `10` | `0.39920520231213874` | `501` | `551` | `"early_stopping"` | `66.41751089900208` |

**`RES-u`**

| Configuration | Trial | Status | n_layers | d | d_hidden_factor | hidden_dropout | residual_dropout | d_embedding | lr | weight_decay | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `RES-u trial 1` | `1` | `"completed"` | `2` | `128` | `2.0` | `0.25` | `0.1` | `192` | `0.001` | `0.0` | `0.3782514450867052` | `14` | `15` | `"maximum"` | `75.71413848499651` |
| `RES-u trial 2` | `2` | `"completed"` | `12` | `132` | `3.130479168686058` | `0.03590023227811617` | `0.0` | `484` | `0.0015459573861767875` | `0.0` | `0.3735549132947977` | `13` | `15` | `"maximum"` | `161.83578258300258` |
| `RES-u trial 3` | `3` | `"completed"` | `3` | `106` | `3.683930287822232` | `0.14484206202702843` | `0.061376951616281816` | `302` | `0.0028772798749728107` | `1.77418272366918e-05` | `0.36127167630057805` | `3` | `9` | `"early_stopping"` | `56.71444895600143` |
| `RES-u trial 4` | `4` | `"completed"` | `6` | `214` | `2.2342619826129786` | `0.35906571027761636` | `0.0` | `456` | `0.003466921825757415` | `1.0765819719525994e-06` | `0.36127167630057805` | `15` | `15` | `"maximum"` | `135.71653557100217` |
| `RES-u trial 5` | `5` | `"completed"` | `14` | `216` | `1.9258592471186684` | `0.39287792961729145` | `0.32659648158787563` | `442` | `0.0002446155549628323` | `0.0` | `0.377528901734104` | `14` | `15` | `"maximum"` | `184.35917414900177` |
| `RES-u trial 6` | `6` | `"completed"` | `4` | `288` | `3.5650580360604645` | `0.15442784278207156` | `0.0` | `141` | `0.0003493373118435067` | `1.116189559640079e-06` | `0.38258670520231214` | `13` | `15` | `"maximum"` | `94.65503486100351` |
| `RES-u trial 7` | `7` | `"completed"` | `8` | `281` | `2.775774866284327` | `0.1627882647085166` | `0.17412297427337764` | `174` | `0.00045716906599527603` | `0.0` | `0.37391618497109824` | `6` | `12` | `"early_stopping"` | `102.2697475889945` |
| `RES-u trial 8` | `8` | `"completed"` | `10` | `235` | `3.780540683737454` | `0.18971705150221702` | `0.0` | `84` | `0.0003774249693509198` | `0.0002462038927745929` | `0.3854768786127168` | `15` | `15` | `"maximum"` | `132.65258197399817` |
| `RES-u trial 9` | `9` | `"completed"` | `11` | `848` | `1.342620080390915` | `0.23408898420368368` | `0.2823336010630874` | `277` | `0.00032972529669487874` | `0.0` | `0.37427745664739887` | `10` | `15` | `"maximum"` | `217.37840233400493` |
| `RES-u trial 10` | `10` | `"completed"` | `14` | `1010` | `3.5329847380799` | `0.49846777509528856` | `0.0` | `405` | `0.005525263577401647` | `0.0` | `0.3522398843930636` | `2` | `8` | `"early_stopping"` | `243.70788334899407` |

**`FTT-u`**

| Configuration | Trial | Status | n_layers | d_token | attention_dropout | ffn_dropout | lr | weight_decay | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FTT-u trial 1` | `1` | `"completed"` | `3` | `192` | `0.2` | `0.1` | `0.0001` | `1e-05` | `0.3645231213872832` | `3` | `9` | `"early_stopping"` | `3926.881690526003` |
| `FTT-u trial 2` | `2` | `"completed"` | `6` | `408` | `0.48566805209520414` | `0.35648520257279626` | `3.020837987267667e-05` | `4.374546306624077e-06` | `0.36741329479768786` | `4` | `10` | `"early_stopping"` | `23426.220827431003` |
| `FTT-u trial 3` | `3` | `"completed"` | `3` | `328` | `0.183259237427422` | `0.4667524517059052` | `3.836328962080732e-05` | `5.11877266143862e-06` | `0.36416184971098264` | `3` | `9` | `"early_stopping"` | `7894.257680406998` |
| `FTT-u trial 4` | `4` | `"completed"` | `3` | `216` | `0.4088941720297744` | `0.26122044047420717` | `0.00028091559146952946` | `1.0594495971607736e-06` | `0.3688583815028902` | `10` | `15` | `"maximum"` | `7808.059355798992` |
| `FTT-u trial 5` | `5` | `"completed"` | `6` | `504` | `0.08874303541781964` | `0.2873813672806639` | `0.00017352298832476001` | `7.724237714716737e-06` | `0.36921965317919075` | `8` | `14` | `"early_stopping"` | `36728.62135195099` |
| `FTT-u trial 6` | `6` | `"completed"` | `2` | `296` | `0.018426853941648558` | `0.28973578677266254` | `0.0002423058075919228` | `7.953327601896765e-06` | `0.3663294797687861` | `14` | `15` | `"maximum"` | `6589.248761873998` |
| `FTT-u trial 7` | `7` | `"completed"` | `6` | `160` | `0.2293649031932407` | `0.42383849674684604` | `0.00010430745790285496` | `9.022443283655203e-06` | `0.3645231213872832` | `9` | `15` | `"early_stopping"` | `15710.010734195996` |
| `FTT-u trial 8` | `8` | `"completed"` | `4` | `232` | `0.4628392203599318` | `0.14296352706450333` | `0.00017313259177615452` | `5.596296579504994e-05` | `0.3684971098265896` | `13` | `15` | `"maximum"` | `11903.103289535007` |
| `FTT-u trial 9` | `9` | `"completed"` | `1` | `328` | `0.3726096168615472` | `0.23834464569527058` | `0.0001575407361166988` | `9.61820691351985e-06` | `0.3616329479768786` | `3` | `9` | `"early_stopping"` | `789.6711085120041` |
| `FTT-u trial 10` | `10` | `"completed"` | `4` | `336` | `0.4414732908681294` | `0.2947031460978327` | `0.00010477662182876941` | `0.0004800367424226514` | `0.368135838150289` | `2` | `8` | `"early_stopping"` | `9857.445310351002` |

**`LGB-b`**

| Configuration | Trial | Status | learning_rate | num_leaves | min_data_in_leaf | feature_fraction | bagging_fraction | lambda_l2 | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LGB-b trial 1` | `1` | `"completed"` | `0.1` | `31` | `20` | `1.0` | `1.0` | `0.0` | `0.3995664739884393` | `594` | `644` | `"early_stopping"` | `60.51887794100912` |
| `LGB-b trial 2` | `2` | `"completed"` | `0.22601761106979648` | `448` | `1133` | `0.6546592048070723` | `0.9773280372088982` | `0.0` | `0.42052023121387283` | `353` | `403` | `"early_stopping"` | `54.26185159099987` |
| `LGB-b trial 3` | `3` | `"completed"` | `0.1290757432936695` | `700` | `39` | `0.6916710240777042` | `0.7896487730540107` | `0.0` | `0.4179913294797688` | `282` | `332` | `"early_stopping"` | `70.58320361902588` |
| `LGB-b trial 4` | `4` | `"completed"` | `0.23475516904426613` | `154` | `28` | `0.9644871396321761` | `0.5072939515473602` | `1.0383864800791283` | `0.3927023121387283` | `173` | `223` | `"early_stopping"` | `34.158886656019604` |
| `LGB-b trial 5` | `5` | `"completed"` | `0.01729915767044926` | `348` | `143` | `0.9049494329519157` | `0.9256684790605736` | `0.0` | `0.4125722543352601` | `697` | `747` | `"early_stopping"` | `133.59161464497447` |
| `LGB-b trial 6` | `6` | `"completed"` | `0.015824883578574592` | `169` | `561` | `0.8881758881483863` | `0.6563225410597064` | `0.1419087232569387` | `0.3952312138728324` | `801` | `851` | `"early_stopping"` | `81.51471998798661` |
| `LGB-b trial 7` | `7` | `"completed"` | `0.08028819960193419` | `431` | `670` | `0.8342502316769134` | `0.5089408954260624` | `2.3035504187312235` | `0.4031791907514451` | `741` | `791` | `"early_stopping"` | `62.734115875005955` |
| `LGB-b trial 8` | `8` | `"completed"` | `0.01218697587169257` | `765` | `1163` | `0.8104764121582517` | `0.5679182667759608` | `3.786140960885953` | `0.38583815028901736` | `768` | `818` | `"early_stopping"` | `54.64198538198252` |
| `LGB-b trial 9` | `9` | `"completed"` | `0.02948061549736706` | `167` | `23` | `0.5389603939382202` | `0.7541846170762744` | `0.283456991321791` | `0.41148843930635837` | `912` | `962` | `"early_stopping"` | `79.32024634900154` |
| `LGB-b trial 10` | `10` | `"completed"` | `0.03261780281441039` | `336` | `1114` | `0.78631223180285` | `0.9756775932427404` | `5.82343521174076` | `0.404985549132948` | `380` | `430` | `"early_stopping"` | `45.656382293993374` |

**`Cat-b`**

| Configuration | Trial | Status | depth | learning_rate | l2_leaf_reg | bagging_temperature | leaf_estimation_iterations | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Cat-b trial 1` | `1` | `"completed"` | `6` | `0.03234` | `3.0` | `1.0` | `10` | `0.381864161849711` | `476` | `526` | `"early_stopping"` | `59.49436142100603` |
| `Cat-b trial 2` | `2` | `"completed"` | `6` | `0.011711449557806689` | `3.940125087194552` | `0.10727525115747005` | `5` | `0.3811416184971098` | `938` | `988` | `"early_stopping"` | `100.40058945902274` |
| `Cat-b trial 3` | `3` | `"completed"` | `9` | `0.04876132783920775` | `6.839996739537394` | `0.08304025522428526` | `10` | `0.3854768786127168` | `226` | `276` | `"early_stopping"` | `46.060885141982` |
| `Cat-b trial 4` | `4` | `"completed"` | `6` | `0.01831704828326315` | `2.9762762356702375` | `0.3571015588198677` | `7` | `0.381864161849711` | `651` | `701` | `"early_stopping"` | `72.822344101005` |
| `Cat-b trial 5` | `5` | `"completed"` | `8` | `0.12744790121234295` | `1.021409401170329` | `0.8560817972120977` | `4` | `0.3898121387283237` | `247` | `297` | `"early_stopping"` | `43.830240627983585` |
| `Cat-b trial 6` | `6` | `"completed"` | `8` | `0.013710329939504001` | `2.4710925897323555` | `0.6679735767142544` | `9` | `0.3793352601156069` | `506` | `556` | `"early_stopping"` | `78.8446739029896` |
| `Cat-b trial 7` | `7` | `"completed"` | `7` | `0.029669263862569156` | `2.257312144928934` | `0.10236368082671032` | `6` | `0.38403179190751446` | `303` | `353` | `"early_stopping"` | `43.87487092000083` |
| `Cat-b trial 8` | `8` | `"completed"` | `9` | `0.1777373665066867` | `6.524576921007237` | `0.5720066363244991` | `7` | `0.38945086705202314` | `182` | `232` | `"early_stopping"` | `39.150841911003226` |
| `Cat-b trial 9` | `9` | `"completed"` | `6` | `0.23525450703996198` | `1.233052876299107` | `0.20837400521592242` | `10` | `0.39378612716763006` | `300` | `350` | `"early_stopping"` | `41.65453742799582` |
| `Cat-b trial 10` | `10` | `"completed"` | `7` | `0.047617053381003464` | `9.490466511636619` | `0.7420573573979781` | `10` | `0.381864161849711` | `268` | `318` | `"early_stopping"` | `44.508607963012764` |

**`RES-b`**

| Configuration | Trial | Status | n_layers | d | d_hidden_factor | hidden_dropout | residual_dropout | d_embedding | lr | weight_decay | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `RES-b trial 1` | `1` | `"completed"` | `2` | `128` | `2.0` | `0.25` | `0.1` | `192` | `0.001` | `0.0` | `0.36560693641618497` | `13` | `15` | `"maximum"` | `95.07170955400215` |
| `RES-b trial 2` | `2` | `"completed"` | `12` | `132` | `3.130479168686058` | `0.03590023227811617` | `0.0` | `484` | `0.0015459573861767875` | `0.0` | `0.36307803468208094` | `13` | `15` | `"maximum"` | `177.1038912080112` |
| `RES-b trial 3` | `3` | `"completed"` | `3` | `106` | `3.683930287822232` | `0.14484206202702843` | `0.061376951616281816` | `302` | `0.0028772798749728107` | `1.77418272366918e-05` | `0.3529624277456647` | `13` | `15` | `"maximum"` | `109.00857923200238` |
| `RES-b trial 4` | `4` | `"completed"` | `6` | `214` | `2.2342619826129786` | `0.35906571027761636` | `0.0` | `456` | `0.003466921825757415` | `1.0765819719525994e-06` | `0.3529624277456647` | `11` | `15` | `"maximum"` | `144.61742443198455` |
| `RES-b trial 5` | `5` | `"completed"` | `14` | `216` | `1.9258592471186684` | `0.39287792961729145` | `0.32659648158787563` | `442` | `0.0002446155549628323` | `0.0` | `0.3634393063583815` | `13` | `15` | `"maximum"` | `187.758534691995` |
| `RES-b trial 6` | `6` | `"completed"` | `4` | `288` | `3.5650580360604645` | `0.15442784278207156` | `0.0` | `141` | `0.0003493373118435067` | `1.116189559640079e-06` | `0.3619942196531792` | `14` | `15` | `"maximum"` | `109.58227491998696` |
| `RES-b trial 7` | `7` | `"completed"` | `8` | `281` | `2.775774866284327` | `0.1627882647085166` | `0.17412297427337764` | `174` | `0.00045716906599527603` | `0.0` | `0.35946531791907516` | `10` | `15` | `"maximum"` | `144.20590329400147` |
| `RES-b trial 8` | `8` | `"completed"` | `10` | `235` | `3.780540683737454` | `0.18971705150221702` | `0.0` | `84` | `0.0003774249693509198` | `0.0002462038927745929` | `0.3598265895953757` | `9` | `15` | `"early_stopping"` | `143.26480442698812` |
| `RES-b trial 9` | `9` | `"completed"` | `11` | `848` | `1.342620080390915` | `0.23408898420368368` | `0.2823336010630874` | `277` | `0.00032972529669487874` | `0.0` | `0.3468208092485549` | `1` | `7` | `"early_stopping"` | `104.31247388699558` |
| `RES-b trial 10` | `10` | `"completed"` | `14` | `1010` | `3.5329847380799` | `0.49846777509528856` | `0.0` | `405` | `0.005525263577401647` | `0.0` | `0.3468208092485549` | `10` | `15` | `"maximum"` | `453.02577790399664` |

**`FTT-b`**

| Configuration | Trial | Status | n_layers | d_token | attention_dropout | ffn_dropout | lr | weight_decay | V1 objective | Best checkpoint | Stopped at | Stopping reason | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FTT-b trial 1` | `1` | `"completed"` | `3` | `192` | `0.2` | `0.1` | `0.0001` | `1e-05` | `0.36488439306358383` | `10` | `15` | `"maximum"` | `6383.398107125977` |
| `FTT-b trial 2` | `2` | `"completed"` | `6` | `408` | `0.48566805209520414` | `0.35648520257279626` | `3.020837987267667e-05` | `4.374546306624077e-06` | `0.3583815028901734` | `4` | `10` | `"early_stopping"` | `22832.504526175995` |
| `FTT-b trial 3` | `3` | `"completed"` | `3` | `328` | `0.183259237427422` | `0.4667524517059052` | `3.836328962080732e-05` | `5.11877266143862e-06` | `0.35585260115606937` | `11` | `15` | `"maximum"` | `12742.153121585987` |
| `FTT-b trial 4` | `4` | `"completed"` | `3` | `216` | `0.4088941720297744` | `0.26122044047420717` | `0.00028091559146952946` | `1.0594495971607736e-06` | `0.36741329479768786` | `15` | `15` | `"maximum"` | `7626.623827333009` |
| `FTT-b trial 5` | `5` | `"completed"` | `6` | `504` | `0.08874303541781964` | `0.2873813672806639` | `0.00017352298832476001` | `7.724237714716737e-06` | `0.36777456647398843` | `10` | `15` | `"maximum"` | `38264.33570257001` |
| `FTT-b trial 6` | `6` | `"completed"` | `2` | `296` | `0.018426853941648558` | `0.28973578677266254` | `0.0002423058075919228` | `7.953327601896765e-06` | `0.3684971098265896` | `15` | `15` | `"maximum"` | `6420.328625792987` |
| `FTT-b trial 7` | `7` | `"completed"` | `6` | `160` | `0.2293649031932407` | `0.42383849674684604` | `0.00010430745790285496` | `9.022443283655203e-06` | `0.35621387283236994` | `4` | `10` | `"early_stopping"` | `10167.703660008963` |
| `FTT-b trial 8` | `8` | `"completed"` | `4` | `232` | `0.4628392203599318` | `0.14296352706450333` | `0.00017313259177615452` | `5.596296579504994e-05` | `0.36307803468208094` | `6` | `12` | `"early_stopping"` | `9292.399159641005` |
| `FTT-b trial 9` | `9` | `"completed"` | `1` | `328` | `0.3726096168615472` | `0.23834464569527058` | `0.0001575407361166988` | `9.61820691351985e-06` | `0.3442919075144509` | `3` | `9` | `"early_stopping"` | `766.1258726769593` |
| `FTT-b trial 10` | `10` | `"completed"` | `4` | `336` | `0.4414732908681294` | `0.2947031460978327` | `0.00010477662182876941` | `0.0004800367424226514` | `0.3670520231213873` | `14` | `15` | `"maximum"` | `17941.25121622195` |

### Development metrics

The V1 objective and the V2 outcomes of every selected main-seed model.

| Configuration | Family | V1 objective | V2 recall (R3) (capacity 0.01) | V2 recall (R3) (capacity 0.02) | V2 recall (R3) (capacity 0.05) | V2 recall (R3) (capacity 0.1) | V2 normalised recall (R3) (capacity 0.01) | V2 normalised recall (R3) (capacity 0.02) | V2 normalised recall (R3) (capacity 0.05) | V2 normalised recall (R3) (capacity 0.1) | V2 Average Precision | V2 ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `0.42015895953757226` | `0.2427000379218809` | `0.40273037542662116` | `0.6006825938566553` | `0.7042093287827076` | `0.8163265306122449` | `0.6772959183673469` | `0.6006825938566553` | `0.7042093287827076` | `0.5325595999956817` | `0.8981804230641399` |
| `Cat-u` | `tree` | `0.40173410404624277` | `0.24421691315889268` | `0.3928706863860447` | `0.5475919605612438` | `0.6697004171406902` | `0.8214285714285714` | `0.6607142857142857` | `0.5475919605612438` | `0.6697004171406902` | `0.5095493793276263` | `0.8872408695638357` |
| `RES-u` | `neural` | `0.3854768786127168` | `0.22639362912400454` | `0.3503981797497156` | `0.49905195297686766` | `0.5991657186196435` | `0.7614795918367347` | `0.5892857142857143` | `0.49905195297686766` | `0.5991657186196435` | `0.4323964899300774` | `0.820644564038209` |
| `FTT-u` | `neural` | `0.36921965317919075` | `0.2169131588926811` | `0.34433067880166857` | `0.4725066363291619` | `0.5832385286310201` | `0.7295918367346939` | `0.5790816326530612` | `0.4725066363291619` | `0.5832385286310201` | `0.4045981211423912` | `0.8022039479964085` |
| `LGB-b` | `tree` | `0.42052023121387283` | `0.24649222601441031` | `0.40576412590064465` | `0.5991657186196435` | `0.7045885475919605` | `0.8290816326530612` | `0.6823979591836735` | `0.5991657186196435` | `0.7045885475919605` | `0.5378443915588387` | `0.9011738129753963` |
| `Cat-b` | `tree` | `0.39378612716763006` | `0.23966628744785742` | `0.37921880925293894` | `0.553280242700038` | `0.6579446340538491` | `0.8061224489795918` | `0.6377551020408163` | `0.553280242700038` | `0.6579446340538491` | `0.49231660278458866` | `0.867350042313778` |
| `RES-b` | `neural` | `0.36560693641618497` | `0.20098596890405765` | `0.3280242700037922` | `0.5051194539249146` | `0.6109215017064846` | `0.6760204081632653` | `0.5516581632653061` | `0.5051194539249146` | `0.6109215017064846` | `0.3942972163464609` | `0.8236564319451786` |
| `FTT-b` | `neural` | `0.3684971098265896` | `0.21008722032612817` | `0.32461130072051575` | `0.48084945013272656` | `0.5847554038680318` | `0.7066326530612245` | `0.5459183673469388` | `0.48084945013272656` | `0.5847554038680318` | `0.39239203726278626` | `0.8123331050757392` |

### Seed variation

V2 Recall@2% under R3 for the main and additional seeds, and the same-seed repeat (DR-015 §4; descriptive only).

| Configuration | Family | V2 Recall@2% by seed (seed 42) | Seed range | Range vs SESOI | Same-seed repeat identical | Same-seed largest score difference |
|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `0.40273037542662116` | — | — | — | — |
| `Cat-u` | `tree` | `0.3928706863860447` | — | — | — | — |
| `RES-u` | `neural` | `0.3503981797497156` | — | — | — | — |
| `FTT-u` | `neural` | `0.34433067880166857` | — | — | — | — |
| `LGB-b` | `tree` | `0.40576412590064465` | — | — | — | — |
| `Cat-b` | `tree` | `0.37921880925293894` | — | — | — | — |
| `RES-b` | `neural` | `0.3280242700037922` | — | — | — | — |
| `FTT-b` | `neural` | `0.32461130072051575` | — | — | — | — |

### CatBoost CPU–GPU comparison

The V2 Recall@2% under R3 of the CatBoost CPU refit minus that of the selected GPU model (the same seed, the GPU-selected settings and tree count), with the percentile interval of the paired two-way bootstrap over relative days and fraud pseudo-entities (DR-013 §7; DR-006). It estimates the difference between the two training procedures, not a hardware effect alone; descriptive only, an interval containing zero is not read as equivalence, and no decision depends on it. The point differences of the other measures are in each record.

| Configuration | Family | CPU − GPU Recall@2% | Interval lower | Interval upper | Interval level | Replicates | Fraud pseudo-entities |
|---|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | — | — | — | — | — | — |
| `Cat-u` | `tree` | — | — | — | — | — | — |
| `RES-u` | `neural` | — | — | — | — | — | — |
| `FTT-u` | `neural` | — | — | — | — | — | — |
| `LGB-b` | `tree` | — | — | — | — | — | — |
| `Cat-b` | `tree` | — | — | — | — | — | — |
| `RES-b` | `neural` | — | — | — | — | — | — |
| `FTT-b` | `neural` | — | — | — | — | — | — |

### RQ3 saturation and break-even ratios

On V2 (the calibrators are fitted on V2, so these are in-sample pre-Test expectations), for the Platt-calibrated probabilities: the saturation ratio rho_sat (from this miss-to-review cost ratio R4 fills the budget on every day; empty when infinite), the break-even ratio rho_BE (the largest ratio at which R4 is strictly cheaper than R3; empty when none or right-censored above 1000) and its case, one column per capacity (DR-007 §4). The ratios are scenario assumptions, never institutional costs; the full curve, the grid and the isotonic check are in each record.

| Configuration | Family |
|---|---|
| `LGB-u` | `tree` |
| `Cat-u` | `tree` |
| `RES-u` | `neural` |
| `FTT-u` | `neural` |
| `LGB-b` | `tree` |
| `Cat-b` | `tree` |
| `RES-b` | `neural` |
| `FTT-b` | `neural` |

### Descriptive contrasts of V2 Recall@2%

Differences of the V2 Recall@2% under R3 of the selected main-seed models: within a family (CatBoost against LightGBM, FT-Transformer against the ResNet-style MLP) and between the families (the mean of the two neural models against the mean of the two tree-based models), for each weighting and for the mean of the two weightings (the point value of the model-family effect δ_F on V2). Descriptive point differences: no interval, no test, no use in selection; the inference on δ_F is the Test evaluation (DR-005). The logistic-regression reference enters no contrast.

| Contrast | Unweighted | Balanced | Mean of the two weightings |
|---|---|---|---|
| `CatBoost − LightGBM` | `-0.009859689040576436` | `-0.026545316647705708` | `-0.018202502844141072` |
| `FT-Transformer − ResNet-style MLP` | `-0.006067500948047033` | `-0.0034129692832764458` | `-0.0047402351156617395` |
| `Neural mean − tree-based mean` | `-0.05043610163064083` | `-0.06617368221463782` | `-0.058304891922639324` |

### Computational cost

Training, scoring and latency on the project machine (protocol Outcomes; descriptive).

| Configuration | Family | Training device | Search (s) | Selected fit (s) | Peak GPU memory (MiB) | V2 scoring (s) | Throughput (rows/s) | Latency median / p95 / p99 (ms) | End-to-end latency median / p95 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `"cpu"` | `425.59792290399855` | `102.96825184100089` | — | `1.5263709970022319` | `50984.98343642611` | `{"median": 0.35860850039171055, "p95": 0.9359290470456467, "p99": 2.7304710421594764}` | `{"median": 68.9162020025833, "p95": 73.4728700510459, "p99": 75.79631397005869}` |
| `Cat-u` | `tree` | `"gpu"` | `592.3427900390016` | `60.79468055399775` | `6788` | `0.44198016399604967` | `176075.77520310518` | `{"median": 0.21917249614489265, "p95": 0.2547962991229724, "p99": 0.30351586021424726}` | `{"median": 6.084648000978632, "p95": 6.544611103890929, "p99": 6.889524748839903}` |
| `RES-u` | `neural` | `"gpu"` | `1410.362605112001` | `132.65258197399817` | `1502` | `1.7681974720035214` | `44012.05251799218` | `{"median": 1.1442670001997612, "p95": 1.4927370484656421, "p99": 1.6913187497993927}` | `{"median": 52.6499990010052, "p95": 56.81005360238487, "p99": 58.56036403842154}` |
| `FTT-u` | `neural` | `"gpu"` | `124637.15376252998` | `36728.62135195099` | `7231` | `325.24583818297833` | `239.27131684377937` | `{"median": 5.201826992561109, "p95": 5.422569897200447, "p99": 5.694377001200333}` | `{"median": 58.60771051084157, "p95": 62.76622683071764, "p99": 66.48985086852916}` |
| `LGB-b` | `tree` | `"cpu"` | `678.9271196219779` | `54.26185159099987` | — | `0.8383575250045396` | `92826.7447704708` | `{"median": 0.17231550009455532, "p95": 0.5015158574678933, "p99": 1.4927055963198623}` | `{"median": 71.49541850958485, "p95": 77.92510613217019, "p99": 82.19400475209112}` |
| `Cat-b` | `tree` | `"gpu"` | `578.8584086410119` | `41.65453742799582` | `6788` | `0.4291022869874723` | `181360.02151457194` | `{"median": 0.1554134942125529, "p95": 0.19480280607240275, "p99": 0.3095612427568991}` | `{"median": 6.059831503080204, "p95": 6.794249446829781, "p99": 7.4735592448268955}` |
| `RES-b` | `neural` | `"gpu"` | `1674.088080829999` | `95.07170955400215` | `722` | `1.8405729089863598` | `42281.40032923668` | `{"median": 0.3861009899992496, "p95": 0.4793801606865599, "p99": 0.712311158131341}` | `{"median": 50.57293549180031, "p95": 55.35943848226452, "p99": 58.28347350412512}` |
| `FTT-b` | `neural` | `"gpu"` | `132442.620223556` | `6420.328625792987` | `7038` | `45.17242033401271` | `1722.7768497806987` | `{"median": 1.2175050214864314, "p95": 1.3563643558882177, "p99": 1.6290574218146499}` | `{"median": 49.66996001894586, "p95": 54.34282708156388, "p99": 56.598681538016535}` |

### H3 configuration selection

The eight configurations ranked by the V2 frauds captured under R3 at 2 % capacity (DR-010): the binding selection on all of V2 and the label-latency check on the fitting days (descriptive; it does not change the H3 configuration).


**Binding selection (all of V2)**

Days 127–154, K = 56; selected `LGB-b`; tie-break steps: none; rule version 1.

| Configuration | Family | Frauds captured | V2 frauds | Recall | Average Precision | Selected |
|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `1062` | `2637` | `0.40273037542662116` | `0.5325595999956817` |  |
| `Cat-u` | `tree` | `1036` | `2637` | `0.3928706863860447` | `0.5095493793276263` |  |
| `RES-u` | `neural` | `924` | `2637` | `0.3503981797497156` | `0.4323964899300774` |  |
| `FTT-u` | `neural` | `908` | `2637` | `0.34433067880166857` | `0.4045981211423912` |  |
| `LGB-b` | `tree` | `1070` | `2637` | `0.40576412590064465` | `0.5378443915588387` | yes |
| `Cat-b` | `tree` | `1000` | `2637` | `0.37921880925293894` | `0.49231660278458866` |  |
| `RES-b` | `neural` | `865` | `2637` | `0.3280242700037922` | `0.3942972163464609` |  |
| `FTT-b` | `neural` | `856` | `2637` | `0.32461130072051575` | `0.39239203726278626` |  |

**Label-latency check**

Days 127–147, K = 56; selected `LGB-u`; tie-break steps: none; rule version 1.

| Configuration | Family | Frauds captured | V2 frauds | Recall | Average Precision | Selected |
|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `789` | `1977` | `0.3990895295902883` | `0.5190365274099409` | yes |
| `Cat-u` | `tree` | `759` | `1977` | `0.38391502276176026` | `0.4954376502611225` |  |
| `RES-u` | `neural` | `675` | `1977` | `0.34142640364188165` | `0.41744889507181826` |  |
| `FTT-u` | `neural` | `671` | `1977` | `0.33940313606474454` | `0.4007387720440098` |  |
| `LGB-b` | `tree` | `788` | `1977` | `0.39858371269600407` | `0.5233150475380669` |  |
| `Cat-b` | `tree` | `745` | `1977` | `0.3768335862417805` | `0.4784869609137185` |  |
| `RES-b` | `neural` | `650` | `1977` | `0.3287809812847749` | `0.4009433489966766` |  |
| `FTT-b` | `neural` | `646` | `1977` | `0.32675771370763784` | `0.38590757294672057` |  |

### Rolling-origin supplement (primary configuration)

The exploratory transfer of the R2 threshold across origins (protocol; DR-001, DR-003): at each origin the primary configuration is refitted on the days before the gap, the R2 threshold (2 % capacity) is set on the scores of the next 28 days (no labels) and applied to the following 28 days against the origin's fixed daily budget K. Retrospective: the configuration was selected on V1 and V2, which lie after the early origins. Descriptive only; the origins overlap and share training data, so no interval or test is given. The band count is the origins whose pooled transfer rate lies strictly inside (0.016, 0.024). Days without an alert are left out of the rate and volume components (DR-003) and counted.

Configuration `LGB-b`.

| Origin | Training days | Training rows | Window days | Transfer days | R2 threshold | K | Window rate | Transfer rate | r / c − 1 | Days above K | Days below K | Overflow | Unused | Zero-alert days | Rate component | Volume component |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `71` | 1–49 | `192688` | 71–98 | 99–126 | `0.12836593422241163` | `65` | `0.020000439811760567` | `0.021886123046087568` | `0.09430615230437844` | `12` | `16` | `332` | `252` | `0` | `0.0333789149767933` | `-0.054475670999658964` |
| `85` | 1–63 | `233881` | 85–112 | 113–140 | `0.10939142411067493` | `66` | `0.020004348771472058` | `0.023871585660075408` | `0.19357928300377036` | `13` | `15` | `366` | `245` | `0` | `0.107655866981066` | `-0.12064484094951941` |
| `99` | 1–77 | `277368` | 99–126 | 127–154 | `0.451322697680012` | `62` | `0.020008524068975846` | `0.013582277505075686` | `-0.3208861247462157` | `2` | `26` | `25` | `704` | `0` | `-0.4323491381204719` | `-0.121003914583418` |

Origins inside the band: 2 of 3.

## Records

### `run.json`

| Field | Value |
|---|---|
| `record` | `"run"` |
| `schema_version` | `3` |
| `started` | `"2026-10-29T01:34:17.893690Z"` |
| `code.package.name` | `"review_capacity"` |
| `code.package.version` | `"0.1.0"` |
| `code.package.location` | `"/home/chathuranga/projects/fraud-detection-review-capacity/src/review_capacity"` |
| `code.package.source_sha256` | `"06540fd34ff853bfdc88ca3bc4ac291f656f392683bb6261f513353667abb5be"` |
| `code.git_commit` | `"6e2ff564e7b5a0dc505b7ee9c9465b4449da0092"` |
| `code.git_uncommitted_changes` | `false` |
| `code.python` | `"3.12.3"` |
| `code.packages.numpy` | `"2.5.3"` |
| `code.packages.pandas` | `"3.0.6"` |
| `code.packages.pyarrow` | `"25.0.1"` |
| `code.packages.PyYAML` | `"6.0.3"` |
| `environment.python` | `"3.12.3"` |
| `environment.platform` | `"Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39"` |
| `environment.packages` | `{"lightgbm": "4.7.0", "catboost": "1.2.10", "torch": "2.11.0+cu128", "numpy": "2.5.3", "pandas": "3.0.6", "scikit-learn": "1.9.1"}` |
| `environment.cpu` | `{"logical_processors": 24, "affinity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "visible_memory_bytes": 58967851008}` |
| `environment.torch_threads` | `{"intraop": 24, "interop": 24}` |
| `environment.gpu` | `{"torch_cuda": "12.8", "cudnn": 91900, "available": true, "driver": "592.01", "name": "NVIDIA GeForce RTX 5060 Laptop GPU", "compute_capability": "12.0", "memory_bytes": 8546484224, "multiprocessors": 26}` |
| `environment.environment_variables` | `{"CUBLAS_WORKSPACE_CONFIG": null, "CUBLASLT_WORKSPACE_SIZE": null, "TORCH_CUBLASLT_UNIFIED_WORKSPACE": null, "CUDA_VISIBLE_DEVICES": null, "CUDA_DEVICE_ORDER": null, "OMP_NUM_THREADS": null, "OMP_PROC_BIND": null, "OMP_PLACES": null, "OMP_DYNAMIC": null, "OMP_THREAD_LIMIT": null, "OMP_SCHEDULE": null, "MKL_NUM_THREADS": null}` |
| `settings.path` | `"config/models.yaml"` |
| `settings.sha256` | `"84e8b74a49c1d1fc7f56b03df352183da8d7555c4975b8eef6ef5bb67e282d1c"` |
| `declaration.path` | `"config/training.yaml"` |
| `declaration.sha256` | `"3b206368e3930880fd6a45b13b2c6a21a52cb4b87b8242813191648b9e0ce4f7"` |
| `plan.path` | `"outputs/search_plan/2026-10-25T130558Z/cost_prediction.json"` |
| `plan.sha256` | `"f02e2f1efa74c317547842b065bf66ce5b20d8e645a2acf0ba0a0c708621d623"` |
| `dataset.manifest.path` | `"data/datapackage.json"` |
| `dataset.manifest.sha256` | `"eb67c7bdca3509c11abba36147a3ed66406db4caa1570035dc62eeb7910bdfeb"` |
| `dataset.processed.development_joined` | `{"path": "data/processed/development_joined.parquet", "sha256": "8c657e5c4094915cf4d1271d52d36fe61e3366489f44174cac845589c0ee8d85"}` |
| `dataset.processed.train` | `{"path": "data/processed/train.parquet", "sha256": "067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"}` |
| `dataset.processed.v1` | `{"path": "data/processed/v1.parquet", "sha256": "1940bf453fe78e6934010472a3ad54c219d132c7b169180d2477cd09bda6cbcb"}` |
| `dataset.processed.v2` | `{"path": "data/processed/v2.parquet", "sha256": "8f6c429fa74200f34ae91ef55725fdc906d2f10ca3e9e2dd421442bf5340a974"}` |
| `configurations` | `["LGB-u", "Cat-u", "RES-u", "FTT-u", "LGB-b", "Cat-b", "RES-b", "FTT-b"]` |
| `contracts.candidates` | `1` |
| `contracts.cost_prediction` | `2` |
| `contracts.run` | `3` |
| `contracts.preprocessing` | `2` |
| `contracts.preflight` | `1` |
| `contracts.fit` | `1` |
| `contracts.checkpoint` | `1` |
| `contracts.trial` | `1` |
| `contracts.configuration` | `1` |
| `contracts.refit` | `1` |
| `contracts.score` | `1` |
| `contracts.evaluation` | `1` |
| `contracts.timing` | `4` |
| `contracts.device_comparison` | `1` |
| `contracts.calibration` | `1` |
| `contracts.cost_analysis` | `2` |
| `contracts.r1_selection` | `1` |
| `contracts.h3_selection` | `1` |
| `contracts.rolling_origin` | `1` |
| `contracts.run_end` | `1` |
| `source.run_id` | `"2026-10-25T134341Z"` |
| `source.run_end.path` | `"outputs/development/2026-10-25T134341Z/records/run_end.json"` |
| `source.run_end.sha256` | `"be21b1fedbc7625a6a2ccb20d1b0fb30b7050582a30471e3156038b60834c80d"` |

### `evaluation-Cat-b-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `745` |
| `counts.false_positives` | `431` |
| `counts.false_negatives` | `1232` |
| `metrics.recall` | `0.3768335862417805` |
| `metrics.precision` | `0.6335034013605442` |
| `metrics.frauds_per_1000_reviews` | `633.5034013605442` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.6335034013605442` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `18.06590200571883` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.2698593314640902` |
| `metrics.fraud_value_exposed` | `229934.055` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `54` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `53` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `17` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.4784869609137185` |
| `metrics.roc_auc` | `0.863916362383431` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-Cat-u-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `759` |
| `counts.false_positives` | `417` |
| `counts.false_negatives` | `1218` |
| `metrics.recall` | `0.38391502276176026` |
| `metrics.precision` | `0.6454081632653061` |
| `metrics.frauds_per_1000_reviews` | `645.4081632653061` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.6454081632653061` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `18.40539546622898` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.2864125793159385` |
| `metrics.fraud_value_exposed` | `224721.148` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `49` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.4954376502611225` |
| `metrics.roc_auc` | `0.8873810985816318` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-FTT-b-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-FTT-b-selected-V2.json"` |
| `score.sha256` | `"edfb3d13260e432480c4d942122b836f6bcc47bc44ce3c217f7c9933af8cfaf5"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `646` |
| `counts.false_positives` | `530` |
| `counts.false_negatives` | `1331` |
| `metrics.recall` | `0.32675771370763784` |
| `metrics.precision` | `0.5493197278911565` |
| `metrics.frauds_per_1000_reviews` | `549.3197278911565` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.5493197278911565` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `15.665198249254178` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.27225380640516633` |
| `metrics.fraud_value_exposed` | `229179.993` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `19` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `13` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-FTT-b-selected-V2.json"` |
| `score.sha256` | `"edfb3d13260e432480c4d942122b836f6bcc47bc44ce3c217f7c9933af8cfaf5"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.38590757294672057` |
| `metrics.roc_auc` | `0.8145738973824882` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-FTT-u-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-FTT-u-selected-V2.json"` |
| `score.sha256` | `"ae97811465ca59a489fb9fab5b294757ee29a15bf810b2bf731951326df83ab8"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `671` |
| `counts.false_positives` | `505` |
| `counts.false_negatives` | `1306` |
| `metrics.recall` | `0.33940313606474454` |
| `metrics.precision` | `0.5705782312925171` |
| `metrics.frauds_per_1000_reviews` | `570.578231292517` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.5705782312925171` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `16.271436571593735` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.29532788961720174` |
| `metrics.fraud_value_exposed` | `221913.561` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `53` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `10` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-FTT-u-selected-V2.json"` |
| `score.sha256` | `"ae97811465ca59a489fb9fab5b294757ee29a15bf810b2bf731951326df83ab8"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.4007387720440098` |
| `metrics.roc_auc` | `0.800720997809131` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-LGB-b-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-LGB-b-selected-V2.json"` |
| `score.sha256` | `"2361a606a8efcc7a06aa710fca8d2981651b6c18290176e0a9bf41c6067e3b48"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `788` |
| `counts.false_positives` | `388` |
| `counts.false_negatives` | `1189` |
| `metrics.recall` | `0.39858371269600407` |
| `metrics.precision` | `0.6700680272108843` |
| `metrics.frauds_per_1000_reviews` | `670.0680272108843` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.6700680272108843` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `19.108631920142866` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.3167561670458464` |
| `metrics.fraud_value_exposed` | `215165.422` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `54` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-LGB-b-selected-V2.json"` |
| `score.sha256` | `"2361a606a8efcc7a06aa710fca8d2981651b6c18290176e0a9bf41c6067e3b48"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.5233150475380669` |
| `metrics.roc_auc` | `0.8995175056140357` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-LGB-u-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-LGB-u-selected-V2.json"` |
| `score.sha256` | `"f51d16e12bcd5de88d84fd3b1f0fa630c3bd784908c8f2a1a564601fb198e3f9"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `789` |
| `counts.false_positives` | `387` |
| `counts.false_negatives` | `1188` |
| `metrics.recall` | `0.3990895295902883` |
| `metrics.precision` | `0.6709183673469388` |
| `metrics.frauds_per_1000_reviews` | `670.9183673469388` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.6709183673469388` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `19.13288145303645` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.29139141069066243` |
| `metrics.fraud_value_exposed` | `223153.227` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-LGB-u-selected-V2.json"` |
| `score.sha256` | `"f51d16e12bcd5de88d84fd3b1f0fa630c3bd784908c8f2a1a564601fb198e3f9"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.5190365274099409` |
| `metrics.roc_auc` | `0.8989880724021255` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-RES-b-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-RES-b-selected-V2.json"` |
| `score.sha256` | `"2f92882c5b1c8bbe1411f6733cd8ca0b2aab1751bc40c8710c3c9a40448a11e6"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `650` |
| `counts.false_positives` | `526` |
| `counts.false_negatives` | `1327` |
| `metrics.recall` | `0.3287809812847749` |
| `metrics.precision` | `0.5527210884353742` |
| `metrics.frauds_per_1000_reviews` | `552.7210884353741` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.5527210884353742` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `15.762196380828508` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.30621593897326893` |
| `metrics.fraud_value_exposed` | `218484.724` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `15` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-RES-b-selected-V2.json"` |
| `score.sha256` | `"2f92882c5b1c8bbe1411f6733cd8ca0b2aab1751bc40c8710c3c9a40448a11e6"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.4009433489966766` |
| `metrics.roc_auc` | `0.8242461834124675` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `evaluation-RES-u-selected-V2-R3-c02-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-RES-u-selected-V2.json"` |
| `score.sha256` | `"6bbb9f64af2487166fd7e4e98bd03ca87e55857765d86eed316737025e6c6e8d"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `21` |
| `counts.transactions` | `56379` |
| `counts.frauds` | `1977` |
| `counts.selected` | `1176` |
| `counts.true_positives` | `675` |
| `counts.false_positives` | `501` |
| `counts.false_negatives` | `1302` |
| `metrics.recall` | `0.34142640364188165` |
| `metrics.precision` | `0.5739795918367347` |
| `metrics.frauds_per_1000_reviews` | `573.9795918367347` |
| `metrics.oracle_ceiling` | `0.5948406676783005` |
| `metrics.normalised_recall` | `0.5739795918367347` |
| `metrics.random_review_recall` | `0.02085883041558027` |
| `metrics.lift` | `16.368434703168067` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5703490136570561` |
| `metrics.amount_weighted_recall` | `0.3423040316905374` |
| `metrics.fraud_value_exposed` | `207119.953` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (21 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-selected-V2-ranking-latency.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"selected"` |
| `seed` | `42` |
| `score.path` | `"source/records/score-RES-u-selected-V2.json"` |
| `score.sha256` | `"6bbb9f64af2487166fd7e4e98bd03ca87e55857765d86eed316737025e6c6e8d"` |
| `partition` | `"V2"` |
| `period` | `"latency"` |
| `days.first` | `127` |
| `days.last` | `147` |
| `policy` | `"ranking"` |
| `capacity` | — |
| `k` | — |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts` | — |
| `metrics.average_precision` | `0.41744889507181826` |
| `metrics.roc_auc` | `0.8206726486985169` |
| `cutoff_ties` | — |

**`daily`** (0 entries)

None.

### `h3_selection-latency.json`

| Field | Value |
|---|---|
| `record` | `"h3_selection"` |
| `schema_version` | `1` |
| `period` | `"latency"` |
| `selection_days.first` | `127` |
| `selection_days.last` | `147` |
| `binding` | `false` |
| `capacity` | `0.02` |
| `k` | `56` |
| `selected` | `"LGB-u"` |
| `tie_break_steps` | `[]` |
| `rule_version` | `"1"` |

**`candidates`** (8 entries)

| `configuration` | `frauds_captured` | `frauds` | `recall` | `average_precision` | `evaluation.path` | `evaluation.sha256` |
|---|---|---|---|---|---|---|
| `"LGB-u"` | `789` | `1977` | `0.3990895295902883` | `0.5190365274099409` | `"records/evaluation-LGB-u-selected-V2-R3-c02-latency.json"` | `"242ac43c0ab0b46f1839b6926daf2d5454d3661f54c7c2c2bdf00a4b42da85a3"` |
| `"Cat-u"` | `759` | `1977` | `0.38391502276176026` | `0.4954376502611225` | `"records/evaluation-Cat-u-selected-V2-R3-c02-latency.json"` | `"d1079b18bcb04a4dc7e31328b1776cc4ab5f944b71356a4c9859fa94e8cd9aba"` |
| `"RES-u"` | `675` | `1977` | `0.34142640364188165` | `0.41744889507181826` | `"records/evaluation-RES-u-selected-V2-R3-c02-latency.json"` | `"cef8144d2a42f463dba3874d5bfb4143b296063d33a948969ceae403f8e46820"` |
| `"FTT-u"` | `671` | `1977` | `0.33940313606474454` | `0.4007387720440098` | `"records/evaluation-FTT-u-selected-V2-R3-c02-latency.json"` | `"c57847eaa843aef4589f94a0219187c69689d16695d66212d73e1bd5cb9fbd74"` |
| `"LGB-b"` | `788` | `1977` | `0.39858371269600407` | `0.5233150475380669` | `"records/evaluation-LGB-b-selected-V2-R3-c02-latency.json"` | `"e923e1513ce1baa35c6e9934498188013302f949667ac4d2136be65fa71af142"` |
| `"Cat-b"` | `745` | `1977` | `0.3768335862417805` | `0.4784869609137185` | `"records/evaluation-Cat-b-selected-V2-R3-c02-latency.json"` | `"b1997539b2634a20bcd2647966490c69227f87ffeb7b009fcfabb48dcbc0f601"` |
| `"RES-b"` | `650` | `1977` | `0.3287809812847749` | `0.4009433489966766` | `"records/evaluation-RES-b-selected-V2-R3-c02-latency.json"` | `"0c1746d9903203c55bf11a2d8459efc07c71e9ea4100c3c327ab3d8983db59ee"` |
| `"FTT-b"` | `646` | `1977` | `0.32675771370763784` | `0.38590757294672057` | `"records/evaluation-FTT-b-selected-V2-R3-c02-latency.json"` | `"0ec8fd73ce425daa397f9fdc66acdfa31f180021f266f4573cc7641fa549ece6"` |

### `h3_selection-primary.json`

| Field | Value |
|---|---|
| `record` | `"h3_selection"` |
| `schema_version` | `1` |
| `period` | `"primary"` |
| `selection_days.first` | `127` |
| `selection_days.last` | `154` |
| `binding` | `true` |
| `capacity` | `0.02` |
| `k` | `56` |
| `selected` | `"LGB-b"` |
| `tie_break_steps` | `[]` |
| `rule_version` | `"1"` |

**`candidates`** (8 entries)

| `configuration` | `frauds_captured` | `frauds` | `recall` | `average_precision` | `evaluation.path` | `evaluation.sha256` |
|---|---|---|---|---|---|---|
| `"LGB-u"` | `1062` | `2637` | `0.40273037542662116` | `0.5325595999956817` | `"source/records/evaluation-LGB-u-selected-V2-R3-c02.json"` | `"159bb216da339400c76aa97391970d19be392f3b3ee52201e81299e6af596972"` |
| `"Cat-u"` | `1036` | `2637` | `0.3928706863860447` | `0.5095493793276263` | `"source/records/evaluation-Cat-u-selected-V2-R3-c02.json"` | `"972a7285499d4632e36900810eafc59b169d7336a1bf02908c24f995f3e9f75b"` |
| `"RES-u"` | `924` | `2637` | `0.3503981797497156` | `0.4323964899300774` | `"source/records/evaluation-RES-u-selected-V2-R3-c02.json"` | `"245598a1c6abcd9cc6dbb49589e699f1660417543e9e29d142802661d8ca7107"` |
| `"FTT-u"` | `908` | `2637` | `0.34433067880166857` | `0.4045981211423912` | `"source/records/evaluation-FTT-u-selected-V2-R3-c02.json"` | `"744b991a8e427e7061cacd796cb63fb3b81a9acd02e6fedefd841e2d1a4ccea8"` |
| `"LGB-b"` | `1070` | `2637` | `0.40576412590064465` | `0.5378443915588387` | `"source/records/evaluation-LGB-b-selected-V2-R3-c02.json"` | `"106f655e054b19137f1f4cbeea19c86e9d1ab274ddfa13b00067c972322fe2a4"` |
| `"Cat-b"` | `1000` | `2637` | `0.37921880925293894` | `0.49231660278458866` | `"source/records/evaluation-Cat-b-selected-V2-R3-c02.json"` | `"756d4d43291f81f94c0ed6b3726b6f935eaf8a59438c8921cd98ed34d0b4caa6"` |
| `"RES-b"` | `865` | `2637` | `0.3280242700037922` | `0.3942972163464609` | `"source/records/evaluation-RES-b-selected-V2-R3-c02.json"` | `"792e87ca3e890cbfa2a96c4fefd8d51c1b3d92909c72ae95c2328c8a3dac8c3a"` |
| `"FTT-b"` | `856` | `2637` | `0.32461130072051575` | `0.39239203726278626` | `"source/records/evaluation-FTT-b-selected-V2-R3-c02.json"` | `"73b1df6f2c9ed63a98c3f62191e038288b61594aa89a3a6cc70c990d1b66ef19"` |

### `rolling_origin-LGB-b-s71.json`

| Field | Value |
|---|---|
| `record` | `"rolling_origin"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `seed` | `42` |
| `origin` | `71` |
| `training.first_day` | `1` |
| `training.last_day` | `49` |
| `training.rows` | `192688` |
| `training.frauds` | `5629` |
| `training.preprocessing_sha256` | — |
| `fit.selected_trial` | `2` |
| `fit.length` | `353` |
| `fit.device` | `"cpu"` |
| `fit.seconds` | `18.768546506995335` |
| `fit.model.path` | `"models/LGB-b/rolling-s71/model.txt"` |
| `fit.model.sha256` | `"e2593d48f4cc9e18b2fb19b7105d6b950f1ecc33dba551dd3655632dcfd9a5bf"` |
| `fit.source_fit.path` | `"source/records/fit-LGB-b-trial2.json"` |
| `fit.source_fit.sha256` | `"6128abcd98e0d1f56511b27829ccf82d8d228bc5d38dee894253ccfa96d51209"` |
| `scores.path` | `"scores/LGB-b-rolling-s71.parquet"` |
| `scores.sha256` | `"e5a0fa0dcf4fd560c3f5baf09f46362c47529cd047c22c5769f99b88c2326fb4"` |
| `window.first_day` | `71` |
| `window.last_day` | `98` |
| `window.transactions` | `90948` |
| `window.alerts` | `1819` |
| `window.rate` | `0.020000439811760567` |
| `threshold.capacity` | `0.02` |
| `threshold.rank` | `1819` |
| `threshold.value` | `0.12836593422241163` |
| `threshold.k` | `65` |
| `transfer.first_day` | `99` |
| `transfer.last_day` | `126` |
| `transfer.transactions` | `86813` |
| `transfer.alerts` | `1900` |
| `transfer.rate` | `0.021886123046087568` |
| `transfer.relative_deviation` | `0.09430615230437844` |
| `transfer.total_overflow` | `332` |
| `transfer.total_unused` | `252` |
| `transfer.days_above_budget` | `12` |
| `transfer.days_below_budget` | `16` |
| `transfer.zero_alert_days` | `0` |
| `transfer.inside_band` | `true` |
| `transfer.rate_component` | `0.0333789149767933` |
| `transfer.volume_component` | `-0.054475670999658964` |
| `transfer.total_component` | `-0.021668347921823216` |
| `transfer.daily` | `[{"day": 99, "transactions": 3301, "selected": 43, "overflow": 0, "unused": 22, "load_ratio": 0.6615384615384615}, {"day": 100, "transactions": 3052, "selected": 35, "overflow": 0, "unused": 30, "load_ratio": 0.5384615384615384}, {"day": 101, "transactions": 2500, "selected": 44, "overflow": 0, "unused": 21, "load_ratio": 0.676923076923077}, {"day": 102, "transactions": 3166, "selected": 46, "overflow": 0, "unused": 19, "load_ratio": 0.7076923076923077}, {"day": 103, "transactions": 3101, "selected": 52, "overflow": 0, "unused": 13, "load_ratio": 0.8}, {"day": 104, "transactions": 3115, "selected": 48, "overflow": 0, "unused": 17, "load_ratio": 0.7384615384615385}, {"day": 105, "transactions": 3034, "selected": 51, "overflow": 0, "unused": 14, "load_ratio": 0.7846153846153846}, {"day": 106, "transactions": 3221, "selected": 91, "overflow": 26, "unused": 0, "load_ratio": 1.4}, {"day": 107, "transactions": 2809, "selected": 104, "overflow": 39, "unused": 0, "load_ratio": 1.6}, {"day": 108, "transactions": 2287, "selected": 32, "overflow": 0, "unused": 33, "load_ratio": 0.49230769230769234}, {"day": 109, "transactions": 3151, "selected": 61, "overflow": 0, "unused": 4, "load_ratio": 0.9384615384615385}, {"day": 110, "transactions": 3018, "selected": 60, "overflow": 0, "unused": 5, "load_ratio": 0.9230769230769231}, {"day": 111, "transactions": 3180, "selected": 44, "overflow": 0, "unused": 21, "load_ratio": 0.676923076923077}, {"day": 112, "transactions": 3225, "selected": 88, "overflow": 23, "unused": 0, "load_ratio": 1.353846153846154}, {"day": 113, "transactions": 3227, "selected": 96, "overflow": 31, "unused": 0, "load_ratio": 1.476923076923077}, {"day": 114, "transactions": 2758, "selected": 76, "overflow": 11, "unused": 0, "load_ratio": 1.1692307692307693}, {"day": 115, "transactions": 2461, "selected": 78, "overflow": 13, "unused": 0, "load_ratio": 1.2}, {"day": 116, "transactions": 3111, "selected": 86, "overflow": 21, "unused": 0, "load_ratio": 1.323076923076923}, {"day": 117, "transactions": 3042, "selected": 133, "overflow": 68, "unused": 0, "load_ratio": 2.046153846153846}, {"day": 118, "transactions": 3179, "selected": 52, "overflow": 0, "unused": 13, "load_ratio": 0.8}, {"day": 119, "transactions": 3300, "selected": 85, "overflow": 20, "unused": 0, "load_ratio": 1.3076923076923077}, {"day": 120, "transactions": 3941, "selected": 109, "overflow": 44, "unused": 0, "load_ratio": 1.676923076923077}, {"day": 121, "transactions": 3017, "selected": 60, "overflow": 0, "unused": 5, "load_ratio": 0.9230769230769231}, {"day": 122, "transactions": 2507, "selected": 43, "overflow": 0, "unused": 22, "load_ratio": 0.6615384615384615}, {"day": 123, "transactions": 3390, "selected": 79, "overflow": 14, "unused": 0, "load_ratio": 1.2153846153846153}, {"day": 124, "transactions": 4202, "selected": 54, "overflow": 0, "unused": 11, "load_ratio": 0.8307692307692308}, {"day": 125, "transactions": 3467, "selected": 87, "overflow": 22, "unused": 0, "load_ratio": 1.3384615384615384}, {"day": 126, "transactions": 3051, "selected": 63, "overflow": 0, "unused": 2, "load_ratio": 0.9692307692307692}]` |

### `rolling_origin-LGB-b-s85.json`

| Field | Value |
|---|---|
| `record` | `"rolling_origin"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `seed` | `42` |
| `origin` | `85` |
| `training.first_day` | `1` |
| `training.last_day` | `63` |
| `training.rows` | `233881` |
| `training.frauds` | `7418` |
| `training.preprocessing_sha256` | — |
| `fit.selected_trial` | `2` |
| `fit.length` | `353` |
| `fit.device` | `"cpu"` |
| `fit.seconds` | `14.823840882978402` |
| `fit.model.path` | `"models/LGB-b/rolling-s85/model.txt"` |
| `fit.model.sha256` | `"4724b9048dfbcc2dd853f4c8f584628688be996df19675cd741ecc9c674463af"` |
| `fit.source_fit.path` | `"source/records/fit-LGB-b-trial2.json"` |
| `fit.source_fit.sha256` | `"6128abcd98e0d1f56511b27829ccf82d8d228bc5d38dee894253ccfa96d51209"` |
| `scores.path` | `"scores/LGB-b-rolling-s85.parquet"` |
| `scores.sha256` | `"1601f0f1ee4dd615d6f63cbb612d174427791750d9ca38c19ec58df33b151352"` |
| `window.first_day` | `85` |
| `window.last_day` | `112` |
| `window.transactions` | `91980` |
| `window.alerts` | `1840` |
| `window.rate` | `0.020004348771472058` |
| `threshold.capacity` | `0.02` |
| `threshold.rank` | `1840` |
| `threshold.value` | `0.10939142411067493` |
| `threshold.k` | `66` |
| `transfer.first_day` | `113` |
| `transfer.last_day` | `140` |
| `transfer.transactions` | `82483` |
| `transfer.alerts` | `1969` |
| `transfer.rate` | `0.023871585660075408` |
| `transfer.relative_deviation` | `0.19357928300377036` |
| `transfer.total_overflow` | `366` |
| `transfer.total_unused` | `245` |
| `transfer.days_above_budget` | `13` |
| `transfer.days_below_budget` | `15` |
| `transfer.zero_alert_days` | `0` |
| `transfer.inside_band` | `true` |
| `transfer.rate_component` | `0.107655866981066` |
| `transfer.volume_component` | `-0.12064484094951941` |
| `transfer.total_component` | `-0.01754479050431414` |
| `transfer.daily` | `[{"day": 113, "transactions": 3227, "selected": 116, "overflow": 50, "unused": 0, "load_ratio": 1.7575757575757576}, {"day": 114, "transactions": 2758, "selected": 84, "overflow": 18, "unused": 0, "load_ratio": 1.2727272727272727}, {"day": 115, "transactions": 2461, "selected": 95, "overflow": 29, "unused": 0, "load_ratio": 1.4393939393939394}, {"day": 116, "transactions": 3111, "selected": 93, "overflow": 27, "unused": 0, "load_ratio": 1.4090909090909092}, {"day": 117, "transactions": 3042, "selected": 145, "overflow": 79, "unused": 0, "load_ratio": 2.196969696969697}, {"day": 118, "transactions": 3179, "selected": 65, "overflow": 0, "unused": 1, "load_ratio": 0.9848484848484849}, {"day": 119, "transactions": 3300, "selected": 99, "overflow": 33, "unused": 0, "load_ratio": 1.5}, {"day": 120, "transactions": 3941, "selected": 119, "overflow": 53, "unused": 0, "load_ratio": 1.803030303030303}, {"day": 121, "transactions": 3017, "selected": 67, "overflow": 1, "unused": 0, "load_ratio": 1.0151515151515151}, {"day": 122, "transactions": 2507, "selected": 48, "overflow": 0, "unused": 18, "load_ratio": 0.7272727272727273}, {"day": 123, "transactions": 3390, "selected": 82, "overflow": 16, "unused": 0, "load_ratio": 1.2424242424242424}, {"day": 124, "transactions": 4202, "selected": 54, "overflow": 0, "unused": 12, "load_ratio": 0.8181818181818182}, {"day": 125, "transactions": 3467, "selected": 99, "overflow": 33, "unused": 0, "load_ratio": 1.5}, {"day": 126, "transactions": 3051, "selected": 71, "overflow": 5, "unused": 0, "load_ratio": 1.0757575757575757}, {"day": 127, "transactions": 3335, "selected": 57, "overflow": 0, "unused": 9, "load_ratio": 0.8636363636363636}, {"day": 128, "transactions": 2507, "selected": 38, "overflow": 0, "unused": 28, "load_ratio": 0.5757575757575758}, {"day": 129, "transactions": 2348, "selected": 87, "overflow": 21, "unused": 0, "load_ratio": 1.3181818181818181}, {"day": 130, "transactions": 2807, "selected": 58, "overflow": 0, "unused": 8, "load_ratio": 0.8787878787878788}, {"day": 131, "transactions": 2822, "selected": 67, "overflow": 1, "unused": 0, "load_ratio": 1.0151515151515151}, {"day": 132, "transactions": 2752, "selected": 51, "overflow": 0, "unused": 15, "load_ratio": 0.7727272727272727}, {"day": 133, "transactions": 2687, "selected": 65, "overflow": 0, "unused": 1, "load_ratio": 0.9848484848484849}, {"day": 134, "transactions": 2937, "selected": 65, "overflow": 0, "unused": 1, "load_ratio": 0.9848484848484849}, {"day": 135, "transactions": 2429, "selected": 43, "overflow": 0, "unused": 23, "load_ratio": 0.6515151515151515}, {"day": 136, "transactions": 2048, "selected": 38, "overflow": 0, "unused": 28, "load_ratio": 0.5757575757575758}, {"day": 137, "transactions": 2914, "selected": 26, "overflow": 0, "unused": 40, "load_ratio": 0.3939393939393939}, {"day": 138, "transactions": 2846, "selected": 40, "overflow": 0, "unused": 26, "load_ratio": 0.6060606060606061}, {"day": 139, "transactions": 2769, "selected": 33, "overflow": 0, "unused": 33, "load_ratio": 0.5}, {"day": 140, "transactions": 2629, "selected": 64, "overflow": 0, "unused": 2, "load_ratio": 0.9696969696969697}]` |

### `rolling_origin-LGB-b-s99.json`

| Field | Value |
|---|---|
| `record` | `"rolling_origin"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `seed` | `42` |
| `origin` | `99` |
| `training.first_day` | `1` |
| `training.last_day` | `77` |
| `training.rows` | `277368` |
| `training.frauds` | `9294` |
| `training.preprocessing_sha256` | — |
| `fit.selected_trial` | `2` |
| `fit.length` | `353` |
| `fit.device` | `"cpu"` |
| `fit.seconds` | `23.252582349989098` |
| `fit.model.path` | `"models/LGB-b/rolling-s99/model.txt"` |
| `fit.model.sha256` | `"04855f585c55df7de5309ba6ad3d6f01be23ae9c06f47eb995987ea6982ef804"` |
| `fit.source_fit.path` | `"source/records/fit-LGB-b-trial2.json"` |
| `fit.source_fit.sha256` | `"6128abcd98e0d1f56511b27829ccf82d8d228bc5d38dee894253ccfa96d51209"` |
| `scores.path` | `"scores/LGB-b-rolling-s99.parquet"` |
| `scores.sha256` | `"34d65bfa3a6f740981f0d49594e2b3c7975259ae125bd437711adbea95eb8a08"` |
| `window.first_day` | `99` |
| `window.last_day` | `126` |
| `window.transactions` | `86813` |
| `window.alerts` | `1737` |
| `window.rate` | `0.020008524068975846` |
| `threshold.capacity` | `0.02` |
| `threshold.rank` | `1737` |
| `threshold.value` | `0.451322697680012` |
| `threshold.k` | `62` |
| `transfer.first_day` | `127` |
| `transfer.last_day` | `154` |
| `transfer.transactions` | `77822` |
| `transfer.alerts` | `1057` |
| `transfer.rate` | `0.013582277505075686` |
| `transfer.relative_deviation` | `-0.3208861247462157` |
| `transfer.total_overflow` | `25` |
| `transfer.total_unused` | `704` |
| `transfer.days_above_budget` | `2` |
| `transfer.days_below_budget` | `26` |
| `transfer.zero_alert_days` | `0` |
| `transfer.inside_band` | `false` |
| `transfer.rate_component` | `-0.4323491381204719` |
| `transfer.volume_component` | `-0.121003914583418` |
| `transfer.total_component` | `-0.5532032943329811` |
| `transfer.daily` | `[{"day": 127, "transactions": 3335, "selected": 37, "overflow": 0, "unused": 25, "load_ratio": 0.5967741935483871}, {"day": 128, "transactions": 2507, "selected": 26, "overflow": 0, "unused": 36, "load_ratio": 0.41935483870967744}, {"day": 129, "transactions": 2348, "selected": 69, "overflow": 7, "unused": 0, "load_ratio": 1.1129032258064515}, {"day": 130, "transactions": 2807, "selected": 39, "overflow": 0, "unused": 23, "load_ratio": 0.6290322580645161}, {"day": 131, "transactions": 2822, "selected": 58, "overflow": 0, "unused": 4, "load_ratio": 0.9354838709677419}, {"day": 132, "transactions": 2752, "selected": 36, "overflow": 0, "unused": 26, "load_ratio": 0.5806451612903226}, {"day": 133, "transactions": 2687, "selected": 41, "overflow": 0, "unused": 21, "load_ratio": 0.6612903225806451}, {"day": 134, "transactions": 2937, "selected": 44, "overflow": 0, "unused": 18, "load_ratio": 0.7096774193548387}, {"day": 135, "transactions": 2429, "selected": 33, "overflow": 0, "unused": 29, "load_ratio": 0.532258064516129}, {"day": 136, "transactions": 2048, "selected": 22, "overflow": 0, "unused": 40, "load_ratio": 0.3548387096774194}, {"day": 137, "transactions": 2914, "selected": 19, "overflow": 0, "unused": 43, "load_ratio": 0.3064516129032258}, {"day": 138, "transactions": 2846, "selected": 30, "overflow": 0, "unused": 32, "load_ratio": 0.4838709677419355}, {"day": 139, "transactions": 2769, "selected": 21, "overflow": 0, "unused": 41, "load_ratio": 0.3387096774193548}, {"day": 140, "transactions": 2629, "selected": 51, "overflow": 0, "unused": 11, "load_ratio": 0.8225806451612904}, {"day": 141, "transactions": 3000, "selected": 42, "overflow": 0, "unused": 20, "load_ratio": 0.6774193548387096}, {"day": 142, "transactions": 2391, "selected": 34, "overflow": 0, "unused": 28, "load_ratio": 0.5483870967741935}, {"day": 143, "transactions": 2064, "selected": 36, "overflow": 0, "unused": 26, "load_ratio": 0.5806451612903226}, {"day": 144, "transactions": 2718, "selected": 24, "overflow": 0, "unused": 38, "load_ratio": 0.3870967741935484}, {"day": 145, "transactions": 2668, "selected": 29, "overflow": 0, "unused": 33, "load_ratio": 0.46774193548387094}, {"day": 146, "transactions": 2990, "selected": 39, "overflow": 0, "unused": 23, "load_ratio": 0.6290322580645161}, {"day": 147, "transactions": 2718, "selected": 34, "overflow": 0, "unused": 28, "load_ratio": 0.5483870967741935}, {"day": 148, "transactions": 3060, "selected": 42, "overflow": 0, "unused": 20, "load_ratio": 0.6774193548387096}, {"day": 149, "transactions": 2289, "selected": 26, "overflow": 0, "unused": 36, "load_ratio": 0.41935483870967744}, {"day": 150, "transactions": 2209, "selected": 27, "overflow": 0, "unused": 35, "load_ratio": 0.43548387096774194}, {"day": 151, "transactions": 3101, "selected": 36, "overflow": 0, "unused": 26, "load_ratio": 0.5806451612903226}, {"day": 152, "transactions": 3896, "selected": 46, "overflow": 0, "unused": 16, "load_ratio": 0.7419354838709677}, {"day": 153, "transactions": 3105, "selected": 80, "overflow": 18, "unused": 0, "load_ratio": 1.2903225806451613}, {"day": 154, "transactions": 3783, "selected": 36, "overflow": 0, "unused": 26, "load_ratio": 0.5806451612903226}]` |

### `run_end.json`

| Field | Value |
|---|---|
| `record` | `"run_end"` |
| `schema_version` | `1` |
| `finished` | `"2026-10-29T01:35:40.550960Z"` |
| `outcome` | `"completed"` |
| `error` | — |

**`records`** (22 entries)

| `path` | `sha256` |
|---|---|
| `"records/evaluation-Cat-b-selected-V2-R3-c02-latency.json"` | `"b1997539b2634a20bcd2647966490c69227f87ffeb7b009fcfabb48dcbc0f601"` |
| `"records/evaluation-Cat-b-selected-V2-ranking-latency.json"` | `"e4e2eda89e8b5d8a86344e93a454b6fe63f3a19fe72e05a8e94aad8d02e916c6"` |
| `"records/evaluation-Cat-u-selected-V2-R3-c02-latency.json"` | `"d1079b18bcb04a4dc7e31328b1776cc4ab5f944b71356a4c9859fa94e8cd9aba"` |
| `"records/evaluation-Cat-u-selected-V2-ranking-latency.json"` | `"37894a28b6db848eba65d9c553de26f2cd26ec9faade803c030138c4871bfb72"` |
| `"records/evaluation-FTT-b-selected-V2-R3-c02-latency.json"` | `"0ec8fd73ce425daa397f9fdc66acdfa31f180021f266f4573cc7641fa549ece6"` |
| `"records/evaluation-FTT-b-selected-V2-ranking-latency.json"` | `"efedd64f848ac8e1e6f737cca9cb5cdb2fb2fb02edb6400e643e28b4bc15e847"` |
| `"records/evaluation-FTT-u-selected-V2-R3-c02-latency.json"` | `"c57847eaa843aef4589f94a0219187c69689d16695d66212d73e1bd5cb9fbd74"` |
| `"records/evaluation-FTT-u-selected-V2-ranking-latency.json"` | `"0e2fd9231a9fa9d09d4722cae073cd0896b5db5f4073d61748d2f6c1925eab51"` |
| `"records/evaluation-LGB-b-selected-V2-R3-c02-latency.json"` | `"e923e1513ce1baa35c6e9934498188013302f949667ac4d2136be65fa71af142"` |
| `"records/evaluation-LGB-b-selected-V2-ranking-latency.json"` | `"b6e973869d66d5955592d1c032f4a2483d2c0ed7b95d2885f9091e5f907d23f3"` |
| `"records/evaluation-LGB-u-selected-V2-R3-c02-latency.json"` | `"242ac43c0ab0b46f1839b6926daf2d5454d3661f54c7c2c2bdf00a4b42da85a3"` |
| `"records/evaluation-LGB-u-selected-V2-ranking-latency.json"` | `"5d0f5749104405ef965dd3be39d37e9ee52496b4a18753bac46c1660bee3383d"` |
| `"records/evaluation-RES-b-selected-V2-R3-c02-latency.json"` | `"0c1746d9903203c55bf11a2d8459efc07c71e9ea4100c3c327ab3d8983db59ee"` |
| `"records/evaluation-RES-b-selected-V2-ranking-latency.json"` | `"9d79995f4458772d1765e1296ae7a6869c73b6a864fdf7594194300bb289977c"` |
| `"records/evaluation-RES-u-selected-V2-R3-c02-latency.json"` | `"cef8144d2a42f463dba3874d5bfb4143b296063d33a948969ceae403f8e46820"` |
| `"records/evaluation-RES-u-selected-V2-ranking-latency.json"` | `"47215f3611aa34391b513c8ccc06272524830e64d8c85f959c24179d64829c62"` |
| `"records/h3_selection-latency.json"` | `"605c2bea374450397f21d1f101743ae1fdaf5bf2c537a8b7c7f6b6d276803c8a"` |
| `"records/h3_selection-primary.json"` | `"cb4144dc9c1f62a21e82f961bed5f88d6723507f91209fb65bfe57e07b502d60"` |
| `"records/rolling_origin-LGB-b-s71.json"` | `"bacfb572cf4c2b5440d2324b0af0ff3f1130bdcea327b64ba7b474961580a1c4"` |
| `"records/rolling_origin-LGB-b-s85.json"` | `"cef086f10de04a058ebb34b65fd47603e6afae85be51f128914fcf4fe56e4c4c"` |
| `"records/rolling_origin-LGB-b-s99.json"` | `"e9b8b23cd899be1259dbcb9ee5fbd32131fd68be9a5b4d0da711f7f36a06887c"` |
| `"records/run.json"` | `"e0f567e2299d85dd745d2f6d124cdb6ce00bd267db23f44fae2fb32a0622c293"` |
