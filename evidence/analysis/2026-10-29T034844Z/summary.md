# Analysis run 2026-10-29T034844Z

Generated from the JSON records of this run directory; not edited by hand.

| Field | Value |
|---|---|
| `run_type` | `"analysis"` |
| `run_id` | `"2026-10-29T034844Z"` |
| `started` | `"2026-10-29T03:48:47.269003Z"` |
| `outcome` | `"completed"` |
| `finished` | `"2026-10-30T01:14:30.429503Z"` |
| `error` | — |
| `records` | `186` |

## Run identity

| Field | Value |
|---|---|
| `identity.run_id` | `"2026-10-29T034844Z"` |
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

| Configuration | Family | V2 Recall@2% by seed (seed 42) | V2 Recall@2% by seed (seed 43) | V2 Recall@2% by seed (seed 44) | Seed range | Range vs SESOI | Same-seed repeat identical | Same-seed largest score difference |
|---|---|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | `0.40273037542662116` | `0.4012135001896094` | `0.40803943875616233` | `0.006825938566552947` | `"below"` | `true` | `0.0` |
| `Cat-u` | `tree` | `0.3928706863860447` | `0.3875616230565036` | `0.3894577171027683` | `0.005309063329541119` | `"below"` | `false` | `0.32890589225931194` |
| `RES-u` | `neural` | `0.3503981797497156` | `0.3223359878649981` | `0.3329541145240804` | `0.02806219188471748` | `"not below"` | `true` | `0.0` |
| `FTT-u` | `neural` | `0.34433067880166857` | `0.3435722411831627` | `0.33447098976109213` | `0.009859689040576436` | `"below"` | `true` | `0.0` |
| `LGB-b` | `tree` | `0.40576412590064465` | `0.40462646947288583` | `0.4023511566173682` | `0.0034129692832764458` | `"below"` | `true` | `0.0` |
| `Cat-b` | `tree` | `0.37921880925293894` | `0.38149412210845657` | `0.38832006067500946` | `0.009101251422070522` | `"below"` | `false` | `0.9526658709760683` |
| `RES-b` | `neural` | `0.3280242700037922` | `0.3029958285930982` | `0.32992036405005687` | `0.026924535456958665` | `"not below"` | `true` | `0.0` |
| `FTT-b` | `neural` | `0.32461130072051575` | `0.32385286310200984` | `0.33560864618885095` | `0.01175578308684111` | `"below"` | `true` | `0.0` |

### CatBoost CPU–GPU comparison

The V2 Recall@2% under R3 of the CatBoost CPU refit minus that of the selected GPU model (the same seed, the GPU-selected settings and tree count), with the percentile interval of the paired two-way bootstrap over relative days and fraud pseudo-entities (DR-013 §7; DR-006). It estimates the difference between the two training procedures, not a hardware effect alone; descriptive only, an interval containing zero is not read as equivalence, and no decision depends on it. The point differences of the other measures are in each record.

| Configuration | Family | CPU − GPU Recall@2% | Interval lower | Interval upper | Interval level | Replicates | Fraud pseudo-entities |
|---|---|---|---|---|---|---|---|
| `LGB-u` | `tree` | — | — | — | — | — | — |
| `Cat-u` | `tree` | `-0.004171406901782328` | `-0.020524653721691113` | `0.011077514009556387` | `0.95` | `10000` | `1353` |
| `RES-u` | `neural` | — | — | — | — | — | — |
| `FTT-u` | `neural` | — | — | — | — | — | — |
| `LGB-b` | `tree` | — | — | — | — | — | — |
| `Cat-b` | `tree` | `-0.012514220705346985` | `-0.036189320063896475` | `0.010899430270002473` | `0.95` | `10000` | `1353` |
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

## Records

### `run.json`

| Field | Value |
|---|---|
| `record` | `"run"` |
| `schema_version` | `3` |
| `started` | `"2026-10-29T03:48:47.269003Z"` |
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
| `environment.cpu` | `{"logical_processors": 24, "affinity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "visible_memory_bytes": 58967834624}` |
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

### `fit-Cat-b-cpu_refit.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `purpose` | `"cpu_refit"` |
| `trial` | `9` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `300` |
| `fitted_iterations` | `300` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-b/cpu_refit/model.cbm"` |
| `model.sha256` | `"b2a2f76a7809d29aae49b097ae5f13fed0e17ce4f643b22eeae802de7bd922d0"` |
| `model_record.path` | `"models/Cat-b/cpu_refit/model_run.json"` |
| `model_record.sha256` | `"cae3fc71fa29cdeb1a708660ef77c39e64af3ab045a2ee4bd0ca1f8d5f89202e"` |
| `fit_seconds` | `56.46051804099989` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `9` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `300` |
| `fitted_iterations` | `300` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-b/same_seed_repeat/model.cbm"` |
| `model.sha256` | `"be63535b500c6e22adecb31bfc96915353ee2999259a407d5fba4932e1f86715"` |
| `model_record.path` | `"models/Cat-b/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"02deef4941aa060505132f4297ce7ae7902b6beff565777b23a2932a1f824384"` |
| `fit_seconds` | `26.732188435002172` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `9` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `300` |
| `fitted_iterations` | `300` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-b/seed_refit-s43/model.cbm"` |
| `model.sha256` | `"34f7288f5f6bc670b1fc2ca7aed09a985306c7eb567658258c1ccbe2f6390273"` |
| `model_record.path` | `"models/Cat-b/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"ca78f2e2fd2d03716e6e1cd8c5f69efa2dc7c03811f86c2e36a51d5efbd5cf0a"` |
| `fit_seconds` | `28.63174036799319` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `9` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `300` |
| `fitted_iterations` | `300` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-b/seed_refit-s44/model.cbm"` |
| `model.sha256` | `"b03705587289c3a250d163fcf024ca8527b622356343da0388078d125ec7185a"` |
| `model_record.path` | `"models/Cat-b/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"b7ee4fa6f4917f3d38bbe5d359c712f58ba46e7a93cd4d6263922c2451b674bf"` |
| `fit_seconds` | `27.03998399199918` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-u-cpu_refit.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `purpose` | `"cpu_refit"` |
| `trial` | `3` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `328` |
| `fitted_iterations` | `328` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-u/cpu_refit/model.cbm"` |
| `model.sha256` | `"7a2e2ff6cefa24ef81555451a10ee652377ebf89c3ee0f0b2de92f728c74ea59"` |
| `model_record.path` | `"models/Cat-u/cpu_refit/model_run.json"` |
| `model_record.sha256` | `"2a16ccff63f327c3ce5069b7f6f41ddd912d03b6cd169680bc749b4c05755939"` |
| `fit_seconds` | `114.49164551600006` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `3` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `328` |
| `fitted_iterations` | `328` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-u/same_seed_repeat/model.cbm"` |
| `model.sha256` | `"b496f23dd6cbe554c410b1f060ab0a896fd22fb821c87b67a5d576afd399057e"` |
| `model_record.path` | `"models/Cat-u/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"b14ecf288b77126e776e57052080e2a0914a60db7fb5e503e17d19e28915746c"` |
| `fit_seconds` | `44.423734879999756` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `3` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `328` |
| `fitted_iterations` | `328` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-u/seed_refit-s43/model.cbm"` |
| `model.sha256` | `"b64e9c826dcae20f5e742fece5cc954ea7fdfe8a86079a1c0837ab06a940007e"` |
| `model_record.path` | `"models/Cat-u/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"975042a079828c37df72a71ba9b8774d2c214856b291778d3b2f6f8470fc1ae9"` |
| `fit_seconds` | `45.78220584000019` |
| `status` | `"completed"` |
| `error` | — |

### `fit-Cat-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `3` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `328` |
| `fitted_iterations` | `328` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/Cat-u/seed_refit-s44/model.cbm"` |
| `model.sha256` | `"bbbfc63b32bea9c8cb774c74fce10b6a514d257ce63a19864ccade60634ec584"` |
| `model_record.path` | `"models/Cat-u/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"a301352df56fca0e18a259ca808615b5aabc5730bbb73e811d525401619550af"` |
| `fit_seconds` | `46.32885931200008` |
| `status` | `"completed"` |
| `error` | — |

### `fit-FTT-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `6` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `15` |
| `fitted_iterations` | `15` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-ft_transformer-trial6.json"` |
| `preflight.sha256` | `"50d92ef4bf0fdecab1e91d5b9af7e2c1edbd7577e3006d08916fc819dd5dedfa"` |
| `model.path` | `"models/FTT-b/same_seed_repeat/model.pt"` |
| `model.sha256` | `"c6708a56cd1a361fa65506a7bc97e93b490c05a83b90be07c672892cd0d8afda"` |
| `model_record.path` | `"models/FTT-b/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"310d800b9f84f777753e5d74327a2298aa59bbf98b67fc20e59587ce33a3a4ff"` |
| `fit_seconds` | `5500.173199875004` |
| `status` | `"completed"` |
| `error` | — |

### `fit-FTT-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `6` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `15` |
| `fitted_iterations` | `15` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-ft_transformer-trial6.json"` |
| `preflight.sha256` | `"50d92ef4bf0fdecab1e91d5b9af7e2c1edbd7577e3006d08916fc819dd5dedfa"` |
| `model.path` | `"models/FTT-b/seed_refit-s43/model.pt"` |
| `model.sha256` | `"1d1eb66c1d317748dfca10cfaae513203c09a296d15931565cbbd34bf5363c86"` |
| `model_record.path` | `"models/FTT-b/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"69f304702cfe163fd6975d04f16cb56fa058757d3d0386bb157f4217ab91a952"` |
| `fit_seconds` | `5459.207633842001` |
| `status` | `"completed"` |
| `error` | — |

### `fit-FTT-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `6` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `15` |
| `fitted_iterations` | `15` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-ft_transformer-trial6.json"` |
| `preflight.sha256` | `"50d92ef4bf0fdecab1e91d5b9af7e2c1edbd7577e3006d08916fc819dd5dedfa"` |
| `model.path` | `"models/FTT-b/seed_refit-s44/model.pt"` |
| `model.sha256` | `"45ac40f91bf588bf3fedeaa8da6056935d9e48f5993e6c9698bfb62782cad61c"` |
| `model_record.path` | `"models/FTT-b/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"ed4ba768038e5d2e416f2edccfa80186a30748c42f051c9ee0f34a87baae3473"` |
| `fit_seconds` | `5516.545576041004` |
| `status` | `"completed"` |
| `error` | — |

### `fit-FTT-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `5` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `8` |
| `fitted_iterations` | `8` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-ft_transformer-trial5.json"` |
| `preflight.sha256` | `"14aa1ec460962ba84cf86b4a02598b9b4d4f9ba95eff996762d2021331dccf1e"` |
| `model.path` | `"models/FTT-u/same_seed_repeat/model.pt"` |
| `model.sha256` | `"55bd3184c2eca118ab928ad6fd1796c7eadb14ba798ec2fa1244f1b3f8a7f337"` |
| `model_record.path` | `"models/FTT-u/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"cf1ba5cdfc78792411f7661b46a5d69c61bc3bf42e6cc66bff9efe06f07c65e2"` |
| `fit_seconds` | `17091.338969356002` |
| `status` | `"completed"` |
| `error` | — |

### `fit-FTT-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `5` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `8` |
| `fitted_iterations` | `8` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-ft_transformer-trial5.json"` |
| `preflight.sha256` | `"14aa1ec460962ba84cf86b4a02598b9b4d4f9ba95eff996762d2021331dccf1e"` |
| `model.path` | `"models/FTT-u/seed_refit-s43/model.pt"` |
| `model.sha256` | `"054ee1e7d436515bc1b3024bd82c31e8ae606e3593ffc712d961eeeddf94810e"` |
| `model_record.path` | `"models/FTT-u/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"f177e01d92f58989a89acd50b661122ba68655aacf451f0438fa4e16c494fbf6"` |
| `fit_seconds` | `17315.212730628` |
| `status` | `"completed"` |
| `error` | — |

### `fit-FTT-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `5` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `8` |
| `fitted_iterations` | `8` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-ft_transformer-trial5.json"` |
| `preflight.sha256` | `"14aa1ec460962ba84cf86b4a02598b9b4d4f9ba95eff996762d2021331dccf1e"` |
| `model.path` | `"models/FTT-u/seed_refit-s44/model.pt"` |
| `model.sha256` | `"03f1c73f4a55f22c365c6052cd9cb44dc99e9abbd8d2b0e7b8dbcba60afcba61"` |
| `model_record.path` | `"models/FTT-u/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"7e98b7faad9574d5b662686dc0a10da2c1144e7acf4a6c9752aef1d5916ce19a"` |
| `fit_seconds` | `17099.087868477996` |
| `status` | `"completed"` |
| `error` | — |

### `fit-LGB-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `2` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `353` |
| `fitted_iterations` | `353` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/LGB-b/same_seed_repeat/model.txt"` |
| `model.sha256` | `"4fcaac44c58f01d29e28f44550309dbc3b740cd87c1f426df47c9dd8e280f40d"` |
| `model_record.path` | `"models/LGB-b/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"8982acf3ba56062ac4bd5241af3d4b95565831ad9be5542f41771dbb0b58baa8"` |
| `fit_seconds` | `23.176023933003307` |
| `status` | `"completed"` |
| `error` | — |

### `fit-LGB-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `2` |
| `seed` | `43` |
| `device` | `"cpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `353` |
| `fitted_iterations` | `353` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/LGB-b/seed_refit-s43/model.txt"` |
| `model.sha256` | `"61579a8a32e9442ef82426bd50304bbae2ab411f8d3d32974577a78eae386130"` |
| `model_record.path` | `"models/LGB-b/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"64f9dce284d4c3092f6948937f8c7f760248218ea81f05a6d8c584333496bd85"` |
| `fit_seconds` | `32.352274363001925` |
| `status` | `"completed"` |
| `error` | — |

### `fit-LGB-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `2` |
| `seed` | `44` |
| `device` | `"cpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `353` |
| `fitted_iterations` | `353` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/LGB-b/seed_refit-s44/model.txt"` |
| `model.sha256` | `"6408c55fc9ff3f7116fc96418d78337b568a5ecd219423c164095fcf575e9dae"` |
| `model_record.path` | `"models/LGB-b/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"c53165a9623fd1b817e262a88dc4f586d8841708b067200e1ac39a738847cd30"` |
| `fit_seconds` | `23.455901957000606` |
| `status` | `"completed"` |
| `error` | — |

### `fit-LGB-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `5` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `843` |
| `fitted_iterations` | `843` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/LGB-u/same_seed_repeat/model.txt"` |
| `model.sha256` | `"6858f463cf2740fbc770e7d3baffa03be9186ca767bc6b5c1bc5d13335e9aff1"` |
| `model_record.path` | `"models/LGB-u/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"a38687f2c5c17bba59392540ff2f458187bd2271b9dfacf02426073b4b7f1f61"` |
| `fit_seconds` | `142.01307739799995` |
| `status` | `"completed"` |
| `error` | — |

### `fit-LGB-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `5` |
| `seed` | `43` |
| `device` | `"cpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `843` |
| `fitted_iterations` | `843` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/LGB-u/seed_refit-s43/model.txt"` |
| `model.sha256` | `"c621c800b8853e8b6c05dcfe24871cf80536b80ca4dfd5f367e5afa882cb01d0"` |
| `model_record.path` | `"models/LGB-u/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"3a9d5745f17c4f3f29e5a680802eb58fa0d0dc7c6c2fe0f228b630ee1b145bfd"` |
| `fit_seconds` | `107.67055811699993` |
| `status` | `"completed"` |
| `error` | — |

### `fit-LGB-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `5` |
| `seed` | `44` |
| `device` | `"cpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `843` |
| `fitted_iterations` | `843` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing` | — |
| `preflight` | — |
| `model.path` | `"models/LGB-u/seed_refit-s44/model.txt"` |
| `model.sha256` | `"864d6f7eaf7f54310d09e2cbdb38e22adf49ce1b2ec791ae4a656bc3b1a83e24"` |
| `model_record.path` | `"models/LGB-u/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"1868b30d4d2c0594ad13f7736448d059f7d424f3824a2efdbc6e539cec2e0029"` |
| `fit_seconds` | `146.76161266100007` |
| `status` | `"completed"` |
| `error` | — |

### `fit-RES-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `1` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `13` |
| `fitted_iterations` | `13` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-resnet-trial1.json"` |
| `preflight.sha256` | `"aaeebe5e8f149d6eea4af6beefc3b8ea9b68e59477da32a8a4ac0b718539011a"` |
| `model.path` | `"models/RES-b/same_seed_repeat/model.pt"` |
| `model.sha256` | `"4eddfb57c7bd3855a625a8a1d72b5bb2f07fd44ba0048d3fb0fac0ba2ddc47d1"` |
| `model_record.path` | `"models/RES-b/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"93924fc81f8286e5243561fd2a1e331edb7cff8c4a763b276de6dce9294ad906"` |
| `fit_seconds` | `57.028692672000034` |
| `status` | `"completed"` |
| `error` | — |

### `fit-RES-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `1` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `13` |
| `fitted_iterations` | `13` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-resnet-trial1.json"` |
| `preflight.sha256` | `"aaeebe5e8f149d6eea4af6beefc3b8ea9b68e59477da32a8a4ac0b718539011a"` |
| `model.path` | `"models/RES-b/seed_refit-s43/model.pt"` |
| `model.sha256` | `"934590972523d1760bddaef0075eaebb36504c47a2495b74de5151a09f9589bd"` |
| `model_record.path` | `"models/RES-b/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"ea755ea0b023f2cc9eda33f2e477b274d7bfe6097effa51fbc046a6a04ac5891"` |
| `fit_seconds` | `57.10429201399529` |
| `status` | `"completed"` |
| `error` | — |

### `fit-RES-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `purpose` | `"seed_refit"` |
| `trial` | `1` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `13` |
| `fitted_iterations` | `13` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `28.528356296058956` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-resnet-trial1.json"` |
| `preflight.sha256` | `"aaeebe5e8f149d6eea4af6beefc3b8ea9b68e59477da32a8a4ac0b718539011a"` |
| `model.path` | `"models/RES-b/seed_refit-s44/model.pt"` |
| `model.sha256` | `"e1aaf8a1213bea33420395130cc45b085bcfd30f7e1e7b53346581033f897761"` |
| `model_record.path` | `"models/RES-b/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"508345403b5a7e143037eac0e527ad1683dc947976823c2d04a61bbbe982d3f2"` |
| `fit_seconds` | `54.34301294400211` |
| `status` | `"completed"` |
| `error` | — |

### `fit-RES-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `purpose` | `"same_seed_repeat"` |
| `trial` | `8` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `fresh_process` | `true` |
| `early_stopping` | `false` |
| `length` | `15` |
| `fitted_iterations` | `15` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-resnet-trial8.json"` |
| `preflight.sha256` | `"91b5fa7668c742113959803f027d34c5261c049b3de0f2f4002aebc95de2ba19"` |
| `model.path` | `"models/RES-u/same_seed_repeat/model.pt"` |
| `model.sha256` | `"9aec2cefb7508a9968137c893dccaf530d072642c24e403ac01a8cf7240d1094"` |
| `model_record.path` | `"models/RES-u/same_seed_repeat/model_run.json"` |
| `model_record.sha256` | `"bc04b2ed42892f335fcd179a29fdcbc2c6a474914adf8c1cff715400405910fc"` |
| `fit_seconds` | `232.60687950800047` |
| `status` | `"completed"` |
| `error` | — |

### `fit-RES-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `8` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `15` |
| `fitted_iterations` | `15` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-resnet-trial8.json"` |
| `preflight.sha256` | `"91b5fa7668c742113959803f027d34c5261c049b3de0f2f4002aebc95de2ba19"` |
| `model.path` | `"models/RES-u/seed_refit-s43/model.pt"` |
| `model.sha256` | `"8a9c8a30e4256e25b3083b710c91285e2ed117d4dfd1f381317dac7da414f29a"` |
| `model_record.path` | `"models/RES-u/seed_refit-s43/model_run.json"` |
| `model_record.sha256` | `"975197d01c44fb7078bf2362e54a17a6ef1554cab080b4463e454984b2810a79"` |
| `fit_seconds` | `128.46033899799977` |
| `status` | `"completed"` |
| `error` | — |

### `fit-RES-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"fit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `purpose` | `"seed_refit"` |
| `trial` | `8` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `fresh_process` | `false` |
| `early_stopping` | `false` |
| `length` | `15` |
| `fitted_iterations` | `15` |
| `train.rows` | `368632` |
| `train.frauds` | `12484` |
| `train.transaction_ids_sha256` | `"f3c088aa2f9527381fd31890a6f47453b2e98e10e4054d4542bfc1a8a48e5c63"` |
| `train.file_sha256` | `"067a670dbadabf2a471ebe34de8cec2fc4e29f07785de0add3eeedb562a4abc4"` |
| `positive_weight` | `1.0` |
| `preprocessing.path` | `"source/records/preprocessing.json"` |
| `preprocessing.sha256` | `"ce3eb698b0c038db7a885e150fcbd344e7e04792381dc907a330f1d65be12aea"` |
| `preflight.path` | `"source/records/preflight-resnet-trial8.json"` |
| `preflight.sha256` | `"91b5fa7668c742113959803f027d34c5261c049b3de0f2f4002aebc95de2ba19"` |
| `model.path` | `"models/RES-u/seed_refit-s44/model.pt"` |
| `model.sha256` | `"853a6822677791213982f19cf4b5e5f0cfa5588a7ba60dcb3eecae9e12b99245"` |
| `model_record.path` | `"models/RES-u/seed_refit-s44/model_run.json"` |
| `model_record.sha256` | `"8c2d8b645053e7c75063a26092be38a69ec99c5aa7593c3cd066f566ca6b9d78"` |
| `fit_seconds` | `162.7048867389999` |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-b-cpu_refit.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `kind` | `"cpu_refit"` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `selected_trial` | `9` |
| `length` | `300` |
| `fit.path` | `"records/fit-Cat-b-cpu_refit.json"` |
| `fit.sha256` | `"2fa268c84d7dd07a277bef40fb9024401af4bcef9e4ddffa285a8d19df0dff40"` |
| `score.path` | `"records/score-Cat-b-cpu_refit-V2.json"` |
| `score.sha256` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `reference_score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `reference_score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `false` |
| `comparison.differing_rows` | `77822` |
| `comparison.max_abs_difference` | `0.9436709348368462` |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `selected_trial` | `9` |
| `length` | `300` |
| `fit.path` | `"records/fit-Cat-b-same_seed_repeat.json"` |
| `fit.sha256` | `"c42a09c43338bb0c7a252b7879aafa7f20c170fbaa65dbb54b48c4227bfa35f4"` |
| `score.path` | `"records/score-Cat-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"6ee29843dda7eca71bf14b8230d6e91a5e505be68319b2ab9bc1f80caef44fe3"` |
| `reference_score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `reference_score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `false` |
| `comparison.differing_rows` | `77822` |
| `comparison.max_abs_difference` | `0.9526658709760683` |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `selected_trial` | `9` |
| `length` | `300` |
| `fit.path` | `"records/fit-Cat-b-seed_refit-s43.json"` |
| `fit.sha256` | `"8230ecb5e5d89f0afaf30273682cc51861470e53c32f906387e957942a7fff20"` |
| `score.path` | `"records/score-Cat-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"209fdd1d5aacc1ea1229110de26984ad0ef4586c175f5b6780588e87f115f5fc"` |
| `reference_score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `reference_score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `selected_trial` | `9` |
| `length` | `300` |
| `fit.path` | `"records/fit-Cat-b-seed_refit-s44.json"` |
| `fit.sha256` | `"1f926d5edbea5363372167644a63f4aae429a3215495d2f3260a4d40f2576369"` |
| `score.path` | `"records/score-Cat-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"cfc3fc337ae49edfe68fd7a23fbe7cf4e85cf5ac5381621810fd699df1a1f6fc"` |
| `reference_score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `reference_score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-u-cpu_refit.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `kind` | `"cpu_refit"` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `selected_trial` | `3` |
| `length` | `328` |
| `fit.path` | `"records/fit-Cat-u-cpu_refit.json"` |
| `fit.sha256` | `"966657c178f5ef32e72c38909ceb8e94b1e26d4fa1d48130d8942aea2751c075"` |
| `score.path` | `"records/score-Cat-u-cpu_refit-V2.json"` |
| `score.sha256` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `reference_score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `reference_score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `false` |
| `comparison.differing_rows` | `77822` |
| `comparison.max_abs_difference` | `0.5786893300969649` |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `selected_trial` | `3` |
| `length` | `328` |
| `fit.path` | `"records/fit-Cat-u-same_seed_repeat.json"` |
| `fit.sha256` | `"b09be1ae6cb53e2c4048dd8fe3ab4559a64d83c7c936aab3a4c78e686bfbec8c"` |
| `score.path` | `"records/score-Cat-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1a9a3415f7d97343ba28ce4f768638ec17e05473242336819881a8132f2d99af"` |
| `reference_score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `reference_score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `false` |
| `comparison.differing_rows` | `77822` |
| `comparison.max_abs_difference` | `0.32890589225931194` |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `selected_trial` | `3` |
| `length` | `328` |
| `fit.path` | `"records/fit-Cat-u-seed_refit-s43.json"` |
| `fit.sha256` | `"3df2d1f16f8d169e0d3368cb4cdba274bd22b70e0f906e9f9af95fb3181f630e"` |
| `score.path` | `"records/score-Cat-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"522d25b74dc8fb4e16391c1e184ce4284e35521e37c1bffa180e4a483dc965ba"` |
| `reference_score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `reference_score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-Cat-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `selected_trial` | `3` |
| `length` | `328` |
| `fit.path` | `"records/fit-Cat-u-seed_refit-s44.json"` |
| `fit.sha256` | `"84f0b0e4d9cd3ddc5590da3bcb594bcc67331e4541a55e0f215f34843d9d18a1"` |
| `score.path` | `"records/score-Cat-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"f6767e8aec9b771cc7f0f2dbf1d2ffb403eb9200bf9e1c01b8de85f906002b9c"` |
| `reference_score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `reference_score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-FTT-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `selected_trial` | `6` |
| `length` | `15` |
| `fit.path` | `"records/fit-FTT-b-same_seed_repeat.json"` |
| `fit.sha256` | `"52e1240d356c1728b9900adc06f7bd1740fb783d2e6bf15301c5b8efbc2d671a"` |
| `score.path` | `"records/score-FTT-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"b6a3d5673c00f4a686f335a9c1f1352e0814f1d1892e32c38299c67243f03d27"` |
| `reference_score.path` | `"source/records/score-FTT-b-selected-V2.json"` |
| `reference_score.sha256` | `"edfb3d13260e432480c4d942122b836f6bcc47bc44ce3c217f7c9933af8cfaf5"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `true` |
| `comparison.differing_rows` | `0` |
| `comparison.max_abs_difference` | `0.0` |
| `status` | `"completed"` |
| `error` | — |

### `refit-FTT-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `selected_trial` | `6` |
| `length` | `15` |
| `fit.path` | `"records/fit-FTT-b-seed_refit-s43.json"` |
| `fit.sha256` | `"f2ad1044a6d60386bdb4bbcf8117192aafde913d5eae3b935c0ef8ea9eef8e6d"` |
| `score.path` | `"records/score-FTT-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"308092a320b1b8096cb71086e7aa0db5a9a5fc9d9963638d9d4ffd71e1ada8a8"` |
| `reference_score.path` | `"source/records/score-FTT-b-selected-V2.json"` |
| `reference_score.sha256` | `"edfb3d13260e432480c4d942122b836f6bcc47bc44ce3c217f7c9933af8cfaf5"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-FTT-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `selected_trial` | `6` |
| `length` | `15` |
| `fit.path` | `"records/fit-FTT-b-seed_refit-s44.json"` |
| `fit.sha256` | `"efa40b3b0f13baa3fac5442695d72d195f438f6a1ae14e4b67f968dce3530fb8"` |
| `score.path` | `"records/score-FTT-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0df888edd13c14014f9978bed7bef834b41d8e81614fcbb5c9bf71a63796c19b"` |
| `reference_score.path` | `"source/records/score-FTT-b-selected-V2.json"` |
| `reference_score.sha256` | `"edfb3d13260e432480c4d942122b836f6bcc47bc44ce3c217f7c9933af8cfaf5"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-FTT-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `selected_trial` | `5` |
| `length` | `8` |
| `fit.path` | `"records/fit-FTT-u-same_seed_repeat.json"` |
| `fit.sha256` | `"bb4be736b9192bc198a08d360aabcd35ef8e6f5d8ac0545d694bb611d6ac209c"` |
| `score.path` | `"records/score-FTT-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"fe96eb55d02b6115e79e7b4a447e4b996341d59031a2cfed12cfc8b4288e7332"` |
| `reference_score.path` | `"source/records/score-FTT-u-selected-V2.json"` |
| `reference_score.sha256` | `"ae97811465ca59a489fb9fab5b294757ee29a15bf810b2bf731951326df83ab8"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `true` |
| `comparison.differing_rows` | `0` |
| `comparison.max_abs_difference` | `0.0` |
| `status` | `"completed"` |
| `error` | — |

### `refit-FTT-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `selected_trial` | `5` |
| `length` | `8` |
| `fit.path` | `"records/fit-FTT-u-seed_refit-s43.json"` |
| `fit.sha256` | `"9332e0e0af09e7bd4c2dfde4643a4c4a157ebdf9dc04f57685733011f29b631e"` |
| `score.path` | `"records/score-FTT-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b10d5e80dd3fa1e3baa334e439817292ecba6dc9abc25e5192235c87eb53bc20"` |
| `reference_score.path` | `"source/records/score-FTT-u-selected-V2.json"` |
| `reference_score.sha256` | `"ae97811465ca59a489fb9fab5b294757ee29a15bf810b2bf731951326df83ab8"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-FTT-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `selected_trial` | `5` |
| `length` | `8` |
| `fit.path` | `"records/fit-FTT-u-seed_refit-s44.json"` |
| `fit.sha256` | `"7f0b5d5f7c8696f2d6edcb57e8fa33edb730a9ef084d88189531938105b90a06"` |
| `score.path` | `"records/score-FTT-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"4521c4c7746e29af794a88c36ed376e851fcc4a2c3d3084d38aad486e32d6103"` |
| `reference_score.path` | `"source/records/score-FTT-u-selected-V2.json"` |
| `reference_score.sha256` | `"ae97811465ca59a489fb9fab5b294757ee29a15bf810b2bf731951326df83ab8"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-LGB-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `selected_trial` | `2` |
| `length` | `353` |
| `fit.path` | `"records/fit-LGB-b-same_seed_repeat.json"` |
| `fit.sha256` | `"e1747e0ff395270a91561af1c47e784cc00197500be9f2ab600e94fed79d57ed"` |
| `score.path` | `"records/score-LGB-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"7fe3d04d2c96421bcfd94b61164a66fed588a0244f45ad0670af89acc6292e77"` |
| `reference_score.path` | `"source/records/score-LGB-b-selected-V2.json"` |
| `reference_score.sha256` | `"2361a606a8efcc7a06aa710fca8d2981651b6c18290176e0a9bf41c6067e3b48"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `true` |
| `comparison.differing_rows` | `0` |
| `comparison.max_abs_difference` | `0.0` |
| `status` | `"completed"` |
| `error` | — |

### `refit-LGB-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"cpu"` |
| `selected_trial` | `2` |
| `length` | `353` |
| `fit.path` | `"records/fit-LGB-b-seed_refit-s43.json"` |
| `fit.sha256` | `"c9e607a828313be32e0a66ce6c4fb34d181cb83c198b7e0c0ee303cf564d6f86"` |
| `score.path` | `"records/score-LGB-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"8a958e7b1aa31e3bfaf9c1a64cae109c10f54c093cbc56c08e4c9cb65b35ada0"` |
| `reference_score.path` | `"source/records/score-LGB-b-selected-V2.json"` |
| `reference_score.sha256` | `"2361a606a8efcc7a06aa710fca8d2981651b6c18290176e0a9bf41c6067e3b48"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-LGB-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"cpu"` |
| `selected_trial` | `2` |
| `length` | `353` |
| `fit.path` | `"records/fit-LGB-b-seed_refit-s44.json"` |
| `fit.sha256` | `"da69d1f9568d654131d22d4bf85cc99bf5b7c3f19eae6103303db34d430a0083"` |
| `score.path` | `"records/score-LGB-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"96df74dd133fc30c2314dbe7094030d2f0035cc49604fe02e69462064bc44eaf"` |
| `reference_score.path` | `"source/records/score-LGB-b-selected-V2.json"` |
| `reference_score.sha256` | `"2361a606a8efcc7a06aa710fca8d2981651b6c18290176e0a9bf41c6067e3b48"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-LGB-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"cpu"` |
| `selected_trial` | `5` |
| `length` | `843` |
| `fit.path` | `"records/fit-LGB-u-same_seed_repeat.json"` |
| `fit.sha256` | `"e011a68b0e4c6b6bea807b882bf5b5e42568d46f5793d256ed9a9d9370c03295"` |
| `score.path` | `"records/score-LGB-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"d66fcf663107e4e2b616e1795573e399c784a323ec8792f802ae4e1aa728bc46"` |
| `reference_score.path` | `"source/records/score-LGB-u-selected-V2.json"` |
| `reference_score.sha256` | `"f51d16e12bcd5de88d84fd3b1f0fa630c3bd784908c8f2a1a564601fb198e3f9"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `true` |
| `comparison.differing_rows` | `0` |
| `comparison.max_abs_difference` | `0.0` |
| `status` | `"completed"` |
| `error` | — |

### `refit-LGB-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"cpu"` |
| `selected_trial` | `5` |
| `length` | `843` |
| `fit.path` | `"records/fit-LGB-u-seed_refit-s43.json"` |
| `fit.sha256` | `"f82579a183112ad78249499475dad460e157f5fd8e8d7d5e40868dc82e18a5d0"` |
| `score.path` | `"records/score-LGB-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"29163846b44103353e020e881b6a9e99e27be338e8b864a7dd3888908ffdeecd"` |
| `reference_score.path` | `"source/records/score-LGB-u-selected-V2.json"` |
| `reference_score.sha256` | `"f51d16e12bcd5de88d84fd3b1f0fa630c3bd784908c8f2a1a564601fb198e3f9"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-LGB-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"cpu"` |
| `selected_trial` | `5` |
| `length` | `843` |
| `fit.path` | `"records/fit-LGB-u-seed_refit-s44.json"` |
| `fit.sha256` | `"d2962980f70ca9043cc3de170bf54bf8a08970b38dde9566ac1a77a5df42a2b2"` |
| `score.path` | `"records/score-LGB-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0c6de8e326b95199ea00e2174e0e50d61a916d7fbaa08660717a280d10e3501f"` |
| `reference_score.path` | `"source/records/score-LGB-u-selected-V2.json"` |
| `reference_score.sha256` | `"f51d16e12bcd5de88d84fd3b1f0fa630c3bd784908c8f2a1a564601fb198e3f9"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-RES-b-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `selected_trial` | `1` |
| `length` | `13` |
| `fit.path` | `"records/fit-RES-b-same_seed_repeat.json"` |
| `fit.sha256` | `"527a0cc34420e873ea666fd6fe9b67277e33bdb91bf0b29c9e825c7d63f56075"` |
| `score.path` | `"records/score-RES-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1bd3f73ace57447940baa252512e545a01f66cf21885e806a83d12e4df000d4a"` |
| `reference_score.path` | `"source/records/score-RES-b-selected-V2.json"` |
| `reference_score.sha256` | `"2f92882c5b1c8bbe1411f6733cd8ca0b2aab1751bc40c8710c3c9a40448a11e6"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `true` |
| `comparison.differing_rows` | `0` |
| `comparison.max_abs_difference` | `0.0` |
| `status` | `"completed"` |
| `error` | — |

### `refit-RES-b-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `selected_trial` | `1` |
| `length` | `13` |
| `fit.path` | `"records/fit-RES-b-seed_refit-s43.json"` |
| `fit.sha256` | `"a944db3a7be956f43a2fc0e00bd3f4573beb494319dba60491e27d22dcb37a67"` |
| `score.path` | `"records/score-RES-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b82d1a99f64a73cbea6c5566c17dfa2de884fac7a6dc4f978f593d1efa8c4d0e"` |
| `reference_score.path` | `"source/records/score-RES-b-selected-V2.json"` |
| `reference_score.sha256` | `"2f92882c5b1c8bbe1411f6733cd8ca0b2aab1751bc40c8710c3c9a40448a11e6"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-RES-b-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `selected_trial` | `1` |
| `length` | `13` |
| `fit.path` | `"records/fit-RES-b-seed_refit-s44.json"` |
| `fit.sha256` | `"79ab05457b2ad09b5db4495213999f80e343278126490cf10c00e818a09fe859"` |
| `score.path` | `"records/score-RES-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"fe64c23cca877970e5f15699b3a4bfae2c893116359aa42b10953f247ef9acf6"` |
| `reference_score.path` | `"source/records/score-RES-b-selected-V2.json"` |
| `reference_score.sha256` | `"2f92882c5b1c8bbe1411f6733cd8ca0b2aab1751bc40c8710c3c9a40448a11e6"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-RES-u-same_seed_repeat.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `kind` | `"same_seed_repeat"` |
| `seed` | `42` |
| `device` | `"gpu"` |
| `selected_trial` | `8` |
| `length` | `15` |
| `fit.path` | `"records/fit-RES-u-same_seed_repeat.json"` |
| `fit.sha256` | `"c68133a5ded03f116949f12a50a24ff033e0b0a65b4cad5dbd9d16835f37c0a1"` |
| `score.path` | `"records/score-RES-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"37b6d7e196e986c237e40975ca94312ad25c17d583e3e347e747dd263a0e63c3"` |
| `reference_score.path` | `"source/records/score-RES-u-selected-V2.json"` |
| `reference_score.sha256` | `"6bbb9f64af2487166fd7e4e98bd03ca87e55857765d86eed316737025e6c6e8d"` |
| `comparison.rows` | `77822` |
| `comparison.identical` | `true` |
| `comparison.differing_rows` | `0` |
| `comparison.max_abs_difference` | `0.0` |
| `status` | `"completed"` |
| `error` | — |

### `refit-RES-u-seed_refit-s43.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `43` |
| `device` | `"gpu"` |
| `selected_trial` | `8` |
| `length` | `15` |
| `fit.path` | `"records/fit-RES-u-seed_refit-s43.json"` |
| `fit.sha256` | `"9874b797c869a9cec68230ea3145f6793adcf140c06fabe4755ad66e85fe6755"` |
| `score.path` | `"records/score-RES-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"7f2eeac675d2f686596ad9c9cfce2534a2e068cab6113235935eaad2bc14d888"` |
| `reference_score.path` | `"source/records/score-RES-u-selected-V2.json"` |
| `reference_score.sha256` | `"6bbb9f64af2487166fd7e4e98bd03ca87e55857765d86eed316737025e6c6e8d"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `refit-RES-u-seed_refit-s44.json`

| Field | Value |
|---|---|
| `record` | `"refit"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `kind` | `"seed_refit"` |
| `seed` | `44` |
| `device` | `"gpu"` |
| `selected_trial` | `8` |
| `length` | `15` |
| `fit.path` | `"records/fit-RES-u-seed_refit-s44.json"` |
| `fit.sha256` | `"d193aadb75dac3731417e0e5d2487e05a52768fb0bccc2acbb665644884598f2"` |
| `score.path` | `"records/score-RES-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"6b8472422a3678635242ad3454c3c8a0f3e5c183a9793d61930140d849d709df"` |
| `reference_score.path` | `"source/records/score-RES-u-selected-V2.json"` |
| `reference_score.sha256` | `"6bbb9f64af2487166fd7e4e98bd03ca87e55857765d86eed316737025e6c6e8d"` |
| `comparison` | — |
| `status` | `"completed"` |
| `error` | — |

### `score-Cat-b-cpu_refit-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-Cat-b-cpu_refit.json"` |
| `fit.sha256` | `"2fa268c84d7dd07a277bef40fb9024401af4bcef9e4ddffa285a8d19df0dff40"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-b-cpu_refit-V2.parquet"` |
| `file.sha256` | `"1c669cd586b06d3720b4b0a88a511ac5cbc4b2e7752c57ddd80c5c05263203c2"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `9.727446255450943e-08` |
| `max_score` | `0.999995156669179` |
| `distinct_scores` | `77775` |
| `seconds` | `0.40710119599680183` |

### `score-Cat-b-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-Cat-b-same_seed_repeat.json"` |
| `fit.sha256` | `"c42a09c43338bb0c7a252b7879aafa7f20c170fbaa65dbb54b48c4227bfa35f4"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-b-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"42e9e6dcf0df67cbb99784690d4d23b73b2dd835b305e6a8e39ef1848a52f46b"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `1.0823559162081483e-08` |
| `max_score` | `0.9999975310265099` |
| `distinct_scores` | `77765` |
| `seconds` | `0.3781689989991719` |

### `score-Cat-b-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-Cat-b-seed_refit-s43.json"` |
| `fit.sha256` | `"8230ecb5e5d89f0afaf30273682cc51861470e53c32f906387e957942a7fff20"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-b-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"179079ee86cc4c159f0aa54a4a58b92b5aac7c7b278aa61e808b7e056ad27647"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `4.430783573296796e-08` |
| `max_score` | `0.9999932856329006` |
| `distinct_scores` | `77740` |
| `seconds` | `0.39530069199827267` |

### `score-Cat-b-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-Cat-b-seed_refit-s44.json"` |
| `fit.sha256` | `"1f926d5edbea5363372167644a63f4aae429a3215495d2f3260a4d40f2576369"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-b-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"2bb163d95a93debfbb50ad26890e60cefb5b79bec57261a8e2af92acc605ea7b"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `2.659164446519066e-08` |
| `max_score` | `0.9999947572572496` |
| `distinct_scores` | `77766` |
| `seconds` | `0.391291492000164` |

### `score-Cat-u-cpu_refit-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-Cat-u-cpu_refit.json"` |
| `fit.sha256` | `"966657c178f5ef32e72c38909ceb8e94b1e26d4fa1d48130d8942aea2751c075"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-u-cpu_refit-V2.parquet"` |
| `file.sha256` | `"85bbdff2fa7097bce9b624062738f6687dcc96c9302f3199bc12166c0ddfcd66"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `0.00023416603689059378` |
| `max_score` | `0.9997679443494037` |
| `distinct_scores` | `77754` |
| `seconds` | `0.46624364399986007` |

### `score-Cat-u-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-Cat-u-same_seed_repeat.json"` |
| `fit.sha256` | `"b09be1ae6cb53e2c4048dd8fe3ab4559a64d83c7c936aab3a4c78e686bfbec8c"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-u-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"d7079492d7fdee7c747a20ce6a02eb4ff3bd84205f4cd987204c2a2483eb38b9"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `0.00013170847607162818` |
| `max_score` | `0.9995763665087634` |
| `distinct_scores` | `77728` |
| `seconds` | `0.42136882199974934` |

### `score-Cat-u-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-Cat-u-seed_refit-s43.json"` |
| `fit.sha256` | `"3df2d1f16f8d169e0d3368cb4cdba274bd22b70e0f906e9f9af95fb3181f630e"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-u-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"1aca976ea5370cd74121089faec2f43cf8c0fb47aa25ce2206163bbb90b1094b"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `0.0001766046395952631` |
| `max_score` | `0.9993971203609662` |
| `distinct_scores` | `77741` |
| `seconds` | `0.44973508800012496` |

### `score-Cat-u-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-Cat-u-seed_refit-s44.json"` |
| `fit.sha256` | `"84f0b0e4d9cd3ddc5590da3bcb594bcc67331e4541a55e0f215f34843d9d18a1"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/Cat-u-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"91e4580fd73f949599b42b6e6410fca799458ca5e4b744e1b754368ce9c4c692"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `0.00015446322001113556` |
| `max_score` | `0.999760139762949` |
| `distinct_scores` | `77722` |
| `seconds` | `0.42979543499996` |

### `score-FTT-b-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-FTT-b-same_seed_repeat.json"` |
| `fit.sha256` | `"52e1240d356c1728b9900adc06f7bd1740fb783d2e6bf15301c5b8efbc2d671a"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/FTT-b-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"44be1e03023808f22d420ca542688bc0798b7b0906ef837c38ef8795441d2554"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `16` |
| `min_score` | `6.463878878888576e-05` |
| `max_score` | `0.9997878977616877` |
| `distinct_scores` | `77441` |
| `seconds` | `40.63539982000657` |

### `score-FTT-b-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-FTT-b-seed_refit-s43.json"` |
| `fit.sha256` | `"f2ad1044a6d60386bdb4bbcf8117192aafde913d5eae3b935c0ef8ea9eef8e6d"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/FTT-b-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"fd41cb1e79619922486765fa8def4b68378765b044737e626b7054631b59183c"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `16` |
| `min_score` | `0.00011517298188771751` |
| `max_score` | `0.9997790960263943` |
| `distinct_scores` | `77653` |
| `seconds` | `40.09472672599804` |

### `score-FTT-b-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-FTT-b-seed_refit-s44.json"` |
| `fit.sha256` | `"efa40b3b0f13baa3fac5442695d72d195f438f6a1ae14e4b67f968dce3530fb8"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/FTT-b-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"6babee2746a959d6ed35e16f2fb8be58f90acb0ba960f1afea53e8dcff3b3e43"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `16` |
| `min_score` | `9.865805419219932e-05` |
| `max_score` | `0.9998280320991477` |
| `distinct_scores` | `77622` |
| `seconds` | `40.13581873799558` |

### `score-FTT-u-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-FTT-u-same_seed_repeat.json"` |
| `fit.sha256` | `"bb4be736b9192bc198a08d360aabcd35ef8e6f5d8ac0545d694bb611d6ac209c"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/FTT-u-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"0c77ea4dd0fdb9480ceddc01117405fa15e46629df7c78580a8d9a25c9c267f6"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `16` |
| `min_score` | `7.818458513161958e-05` |
| `max_score` | `0.9978460756440296` |
| `distinct_scores` | `77520` |
| `seconds` | `293.54902768100146` |

### `score-FTT-u-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-FTT-u-seed_refit-s43.json"` |
| `fit.sha256` | `"9332e0e0af09e7bd4c2dfde4643a4c4a157ebdf9dc04f57685733011f29b631e"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/FTT-u-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"119a2e95c23f806d06b9643fdd76ca401c92c6185c466f3c01f7ad6a99e96620"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `16` |
| `min_score` | `0.0001353123765799906` |
| `max_score` | `0.9987954634986487` |
| `distinct_scores` | `76358` |
| `seconds` | `295.72600490600234` |

### `score-FTT-u-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-FTT-u-seed_refit-s44.json"` |
| `fit.sha256` | `"7f0b5d5f7c8696f2d6edcb57e8fa33edb730a9ef084d88189531938105b90a06"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/FTT-u-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"228049e72e78fc4c4acd9ebad9b5a26be21787d8667df27e22c3ebe52ecdf06a"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `16` |
| `min_score` | `4.84452836546662e-05` |
| `max_score` | `0.9983991652304014` |
| `distinct_scores` | `76492` |
| `seconds` | `295.40343353500066` |

### `score-LGB-b-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-LGB-b-same_seed_repeat.json"` |
| `fit.sha256` | `"e1747e0ff395270a91561af1c47e784cc00197500be9f2ab600e94fed79d57ed"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/LGB-b-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"b9b78881c4d582b6e60b43a12b54fdaf242f1642257e67a94e2ecef7c8322b81"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `1.8458923429936554e-14` |
| `max_score` | `0.9999999993904389` |
| `distinct_scores` | `77817` |
| `seconds` | `0.8135061989960377` |

### `score-LGB-b-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-LGB-b-seed_refit-s43.json"` |
| `fit.sha256` | `"c9e607a828313be32e0a66ce6c4fb34d181cb83c198b7e0c0ee303cf564d6f86"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/LGB-b-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"af3a3284480baf85d3c9ea3e4b345c42894614582d9a57f65ddbbf1e4bd786f3"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `3.3328624389728706e-15` |
| `max_score` | `0.9999999980682961` |
| `distinct_scores` | `77818` |
| `seconds` | `1.263732818995777` |

### `score-LGB-b-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-LGB-b-seed_refit-s44.json"` |
| `fit.sha256` | `"da69d1f9568d654131d22d4bf85cc99bf5b7c3f19eae6103303db34d430a0083"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/LGB-b-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"6e634eef6e19ca07e676b1d537bc194f6feab461040b0cfef8477fdaebcada44"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `2.7572092685717845e-14` |
| `max_score` | `0.9999999984033827` |
| `distinct_scores` | `77818` |
| `seconds` | `0.664305174999754` |

### `score-LGB-u-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-LGB-u-same_seed_repeat.json"` |
| `fit.sha256` | `"e011a68b0e4c6b6bea807b882bf5b5e42568d46f5793d256ed9a9d9370c03295"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/LGB-u-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"ea743cb7577f2e393e0bf3f03d8a4f5e33b8c97cf1ac8bd03f8d1ff86977c61e"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `3.297746237233719e-06` |
| `max_score` | `0.9998395604333066` |
| `distinct_scores` | `77818` |
| `seconds` | `1.591358467000191` |

### `score-LGB-u-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-LGB-u-seed_refit-s43.json"` |
| `fit.sha256` | `"f82579a183112ad78249499475dad460e157f5fd8e8d7d5e40868dc82e18a5d0"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/LGB-u-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"c7c209c779023a2b5daf29044e604a8a890356e79fa9067176038e97196d859a"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `3.075731366951141e-06` |
| `max_score` | `0.999800840065245` |
| `distinct_scores` | `77820` |
| `seconds` | `1.5509410009999556` |

### `score-LGB-u-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-LGB-u-seed_refit-s44.json"` |
| `fit.sha256` | `"d2962980f70ca9043cc3de170bf54bf8a08970b38dde9566ac1a77a5df42a2b2"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/LGB-u-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"bba3238ea0fc5f7b8a00d4f3cf7a26b35f078949479aa6f495cacc697e487cdf"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"cpu"` |
| `batch_rows` | — |
| `min_score` | `2.838896155668651e-06` |
| `max_score` | `0.9997652637418751` |
| `distinct_scores` | `77819` |
| `seconds` | `1.2864506560003974` |

### `score-RES-b-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-RES-b-same_seed_repeat.json"` |
| `fit.sha256` | `"527a0cc34420e873ea666fd6fe9b67277e33bdb91bf0b29c9e825c7d63f56075"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/RES-b-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"0d8c9141126aebded6e15f635a6734a8fc77ef50b90d160c94e3d0483c039a28"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `4096` |
| `min_score` | `2.4677645005297896e-15` |
| `max_score` | `0.9999996563370284` |
| `distinct_scores` | `77755` |
| `seconds` | `0.17235769599938067` |

### `score-RES-b-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-RES-b-seed_refit-s43.json"` |
| `fit.sha256` | `"a944db3a7be956f43a2fc0e00bd3f4573beb494319dba60491e27d22dcb37a67"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/RES-b-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"bf52445b7860777414f2276abd86cb43cfbcac87da7b4435247127a9db61fb48"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `4096` |
| `min_score` | `4.17562082640781e-17` |
| `max_score` | `0.999999894760808` |
| `distinct_scores` | `77753` |
| `seconds` | `0.18036172099527903` |

### `score-RES-b-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-RES-b-seed_refit-s44.json"` |
| `fit.sha256` | `"79ab05457b2ad09b5db4495213999f80e343278126490cf10c00e818a09fe859"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/RES-b-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"06eba31f9ce2ae830a5904dd16a88efec17b6259d229a9f591ea51f9bed9b4e5"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `4096` |
| `min_score` | `1.2150683254465553e-17` |
| `max_score` | `0.9999999237291655` |
| `distinct_scores` | `77754` |
| `seconds` | `0.15852322799764806` |

### `score-RES-u-same_seed_repeat-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `fit.path` | `"records/fit-RES-u-same_seed_repeat.json"` |
| `fit.sha256` | `"c68133a5ded03f116949f12a50a24ff033e0b0a65b4cad5dbd9d16835f37c0a1"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/RES-u-same_seed_repeat-V2.parquet"` |
| `file.sha256` | `"642cc56bd824739186f4336a1adada2e13f422fa8fb232fc6b7897c44c32c65f"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `4096` |
| `min_score` | `4.020007788073937e-07` |
| `max_score` | `0.999999999428451` |
| `distinct_scores` | `77583` |
| `seconds` | `0.3812706380003874` |

### `score-RES-u-seed_refit-s43-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `fit.path` | `"records/fit-RES-u-seed_refit-s43.json"` |
| `fit.sha256` | `"9874b797c869a9cec68230ea3145f6793adcf140c06fabe4755ad66e85fe6755"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/RES-u-seed_refit-s43-V2.parquet"` |
| `file.sha256` | `"96af4dd1c3c9c55525983c21111829bfa5a77b3f0126092becc4a48bc6e4636d"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `4096` |
| `min_score` | `3.0603958137315244e-10` |
| `max_score` | `0.9999999414153983` |
| `distinct_scores` | `77652` |
| `seconds` | `0.3363121500001398` |

### `score-RES-u-seed_refit-s44-V2.json`

| Field | Value |
|---|---|
| `record` | `"score"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `fit.path` | `"records/fit-RES-u-seed_refit-s44.json"` |
| `fit.sha256` | `"d193aadb75dac3731417e0e5d2487e05a52768fb0bccc2acbb665644884598f2"` |
| `partition` | `"V2"` |
| `first_day` | `127` |
| `last_day` | `154` |
| `rows` | `77822` |
| `file.path` | `"scores/RES-u-seed_refit-s44-V2.parquet"` |
| `file.sha256` | `"775ba83732eb93795ef86c4834acf5fa80473b63722499101f5cfafc83eb7a88"` |
| `format` | `"parquet"` |
| `columns` | `["TransactionID", "score"]` |
| `score_dtype` | `"float64"` |
| `device` | `"gpu"` |
| `batch_rows` | `4096` |
| `min_score` | `1.5804225393719587e-07` |
| `max_score` | `0.9999999645481489` |
| `distinct_scores` | `77628` |
| `seconds` | `0.3963505910001004` |

### `evaluation-Cat-b-cpu_refit-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-cpu_refit-V2.json"` |
| `score.sha256` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `616` |
| `counts.false_positives` | `168` |
| `counts.false_negatives` | `2021` |
| `metrics.recall` | `0.2335987864998104` |
| `metrics.precision` | `0.7857142857142857` |
| `metrics.frauds_per_1000_reviews` | `785.7142857142857` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7857142857142857` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `23.187659136464596` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.1407712571732433` |
| `metrics.fraud_value_exposed` | `346065.309` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-cpu_refit-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-cpu_refit-V2.json"` |
| `score.sha256` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `967` |
| `counts.false_positives` | `601` |
| `counts.false_negatives` | `1670` |
| `metrics.recall` | `0.36670458854759197` |
| `metrics.precision` | `0.6167091836734694` |
| `metrics.frauds_per_1000_reviews` | `616.7091836734694` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6167091836734694` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `18.20005388389713` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2529412900683431` |
| `metrics.fraud_value_exposed` | `300887.401` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `53` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `49` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `17` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-cpu_refit-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-cpu_refit-V2.json"` |
| `score.sha256` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1418` |
| `counts.false_positives` | `2474` |
| `counts.false_negatives` | `1219` |
| `metrics.recall` | `0.5377322715206674` |
| `metrics.precision` | `0.3643371017471737` |
| `metrics.frauds_per_1000_reviews` | `364.3371017471737` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5377322715206674` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `10.752158487739306` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4118186741631036` |
| `metrics.fraud_value_exposed` | `236897.513` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `48` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `82` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `55` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `56` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `47` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `79` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `57` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `50` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `40` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `39` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `60` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `60` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `57` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `46` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `47` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `37` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `44` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `31` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `44` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `39` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `32` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `58` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `58` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `89` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-cpu_refit-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-cpu_refit-V2.json"` |
| `score.sha256` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1733` |
| `counts.false_positives` | `6051` |
| `counts.false_negatives` | `904` |
| `metrics.recall` | `0.6571861964353432` |
| `metrics.precision` | `0.22263617677286743` |
| `metrics.frauds_per_1000_reviews` | `222.63617677286743` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6571861964353432` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.570342263488088` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5804484180616477` |
| `metrics.fraud_value_exposed` | `168979.738` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `50` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `97` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `62` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `71` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `65` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `98` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `62` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `58` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `52` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `45` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `74` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `35` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `78` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `67` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `67` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `53` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `47` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `64` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `43` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `52` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `47` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `75` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `72` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `99` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `65` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"6ee29843dda7eca71bf14b8230d6e91a5e505be68319b2ab9bc1f80caef44fe3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `606` |
| `counts.false_positives` | `178` |
| `counts.false_negatives` | `2031` |
| `metrics.recall` | `0.229806598407281` |
| `metrics.precision` | `0.7729591836734694` |
| `metrics.frauds_per_1000_reviews` | `772.9591836734694` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7729591836734694` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `22.811236098534977` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.15385992622296396` |
| `metrics.fraud_value_exposed` | `340793.681` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"6ee29843dda7eca71bf14b8230d6e91a5e505be68319b2ab9bc1f80caef44fe3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `993` |
| `counts.false_positives` | `575` |
| `counts.false_negatives` | `1644` |
| `metrics.recall` | `0.37656427758816835` |
| `metrics.precision` | `0.6332908163265306` |
| `metrics.frauds_per_1000_reviews` | `633.2908163265306` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6332908163265306` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `18.68940383320564` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.26486367939054856` |
| `metrics.fraud_value_exposed` | `296085.507` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `16` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"6ee29843dda7eca71bf14b8230d6e91a5e505be68319b2ab9bc1f80caef44fe3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1440` |
| `counts.false_positives` | `2452` |
| `counts.false_negatives` | `1197` |
| `metrics.recall` | `0.5460750853242321` |
| `metrics.precision` | `0.3699897225077081` |
| `metrics.frauds_per_1000_reviews` | `369.9897225077081` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5460750853242321` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `10.918976179368547` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4532187343136551` |
| `metrics.fraud_value_exposed` | `220223.112` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `43` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `85` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `59` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `64` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `56` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `84` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `55` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `55` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `39` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `37` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `62` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `25` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `52` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `54` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `44` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `43` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `38` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `45` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `39` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `46` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `35` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `58` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `62` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `89` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `43` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"6ee29843dda7eca71bf14b8230d6e91a5e505be68319b2ab9bc1f80caef44fe3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1737` |
| `counts.false_positives` | `6047` |
| `counts.false_negatives` | `900` |
| `metrics.recall` | `0.658703071672355` |
| `metrics.precision` | `0.22315005138746147` |
| `metrics.frauds_per_1000_reviews` | `223.15005138746145` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.658703071672355` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.585507508181656` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6140195585726548` |
| `metrics.fraud_value_exposed` | `155458.534` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `53` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `100` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `68` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `75` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `68` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `99` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `61` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `63` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `48` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `77` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `33` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `75` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `65` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `65` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `50` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `55` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `55` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `45` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `54` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `46` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `79` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `71` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `98` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `60` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"209fdd1d5aacc1ea1229110de26984ad0ef4586c175f5b6780588e87f115f5fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `616` |
| `counts.false_positives` | `168` |
| `counts.false_negatives` | `2021` |
| `metrics.recall` | `0.2335987864998104` |
| `metrics.precision` | `0.7857142857142857` |
| `metrics.frauds_per_1000_reviews` | `785.7142857142857` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7857142857142857` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `23.187659136464596` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.15476493549586778` |
| `metrics.fraud_value_exposed` | `340429.177` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"209fdd1d5aacc1ea1229110de26984ad0ef4586c175f5b6780588e87f115f5fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1006` |
| `counts.false_positives` | `562` |
| `counts.false_negatives` | `1631` |
| `metrics.recall` | `0.38149412210845657` |
| `metrics.precision` | `0.6415816326530612` |
| `metrics.frauds_per_1000_reviews` | `641.5816326530612` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6415816326530612` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `18.93407880785989` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.29149817250714594` |
| `metrics.fraud_value_exposed` | `285358.126` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"209fdd1d5aacc1ea1229110de26984ad0ef4586c175f5b6780588e87f115f5fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1457` |
| `counts.false_positives` | `2435` |
| `counts.false_negatives` | `1180` |
| `metrics.recall` | `0.552521805081532` |
| `metrics.precision` | `0.37435765673175747` |
| `metrics.frauds_per_1000_reviews` | `374.35765673175746` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.552521805081532` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.04788075926387` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4799564321151031` |
| `metrics.fraud_value_exposed` | `209454.164` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `46` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `84` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `55` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `59` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `51` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `85` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `60` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `55` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `42` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `63` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `56` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `54` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `46` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `46` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `37` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `46` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `40` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `46` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `35` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `61` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `64` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"209fdd1d5aacc1ea1229110de26984ad0ef4586c175f5b6780588e87f115f5fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1715` |
| `counts.false_positives` | `6069` |
| `counts.false_negatives` | `922` |
| `metrics.recall` | `0.6503602578687903` |
| `metrics.precision` | `0.22032374100719423` |
| `metrics.frauds_per_1000_reviews` | `220.32374100719426` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6503602578687903` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.502098662367034` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6064195131424327` |
| `metrics.fraud_value_exposed` | `158519.549` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `50` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `96` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `67` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `60` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `99` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `65` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `64` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `46` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `39` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `76` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `39` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `78` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `70` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `62` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `52` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `61` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `41` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `53` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `46` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `56` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `43` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `76` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `74` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `94` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `60` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"cfc3fc337ae49edfe68fd7a23fbe7cf4e85cf5ac5381621810fd699df1a1f6fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `628` |
| `counts.false_positives` | `156` |
| `counts.false_negatives` | `2009` |
| `metrics.recall` | `0.23814941221084565` |
| `metrics.precision` | `0.8010204081632653` |
| `metrics.frauds_per_1000_reviews` | `801.0204081632653` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8010204081632653` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `23.63936678198014` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.15686630920690667` |
| `metrics.fraud_value_exposed` | `339582.822` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"cfc3fc337ae49edfe68fd7a23fbe7cf4e85cf5ac5381621810fd699df1a1f6fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1024` |
| `counts.false_positives` | `544` |
| `counts.false_negatives` | `1613` |
| `metrics.recall` | `0.38832006067500946` |
| `metrics.precision` | `0.6530612244897959` |
| `metrics.frauds_per_1000_reviews` | `653.0612244897959` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6530612244897959` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.27285954199655` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.3028614559673104` |
| `metrics.fraud_value_exposed` | `280781.419` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `19` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"cfc3fc337ae49edfe68fd7a23fbe7cf4e85cf5ac5381621810fd699df1a1f6fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1490` |
| `counts.false_positives` | `2402` |
| `counts.false_negatives` | `1147` |
| `metrics.recall` | `0.5650360257868791` |
| `metrics.precision` | `0.3828365878725591` |
| `metrics.frauds_per_1000_reviews` | `382.8365878725591` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5650360257868791` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.298107296707734` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.5055292008602067` |
| `metrics.fraud_value_exposed` | `199154.406` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `46` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `84` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `56` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `62` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `61` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `85` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `59` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `57` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `38` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `36` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `60` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `28` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `67` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `66` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `54` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `43` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `46` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `39` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `47` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `40` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `47` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `41` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `59` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `63` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `89` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `51` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-b-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"cfc3fc337ae49edfe68fd7a23fbe7cf4e85cf5ac5381621810fd699df1a1f6fc"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1752` |
| `counts.false_positives` | `6032` |
| `counts.false_negatives` | `885` |
| `metrics.recall` | `0.664391353811149` |
| `metrics.precision` | `0.2250770811921891` |
| `metrics.frauds_per_1000_reviews` | `225.0770811921891` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.664391353811149` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.642377175782533` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6469834854925651` |
| `metrics.fraud_value_exposed` | `142181.893` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `48` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `97` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `75` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `70` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `97` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `64` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `46` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `72` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `36` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `76` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `77` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `65` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `58` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `57` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `57` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `45` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `44` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `72` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `78` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `98` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `65` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-cpu_refit-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-cpu_refit-V2.json"` |
| `score.sha256` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `641` |
| `counts.false_positives` | `143` |
| `counts.false_negatives` | `1996` |
| `metrics.recall` | `0.24307925673113387` |
| `metrics.precision` | `0.8176020408163265` |
| `metrics.frauds_per_1000_reviews` | `817.6020408163265` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8176020408163265` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.128716731288648` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.17128276498596726` |
| `metrics.fraud_value_exposed` | `333776.411` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-cpu_refit-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-cpu_refit-V2.json"` |
| `score.sha256` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1025` |
| `counts.false_positives` | `543` |
| `counts.false_negatives` | `1612` |
| `metrics.recall` | `0.3886992794842624` |
| `metrics.precision` | `0.6536989795918368` |
| `metrics.frauds_per_1000_reviews` | `653.6989795918367` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6536989795918368` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.291680693893028` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2764100036228775` |
| `metrics.fraud_value_exposed` | `291435.078` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-cpu_refit-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-cpu_refit-V2.json"` |
| `score.sha256` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1467` |
| `counts.false_positives` | `2425` |
| `counts.false_negatives` | `1170` |
| `metrics.recall` | `0.5563139931740614` |
| `metrics.precision` | `0.37692702980472764` |
| `metrics.frauds_per_1000_reviews` | `376.9270298047276` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5563139931740614` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.123706982731708` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4431062691889034` |
| `metrics.fraud_value_exposed` | `224296.036` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `49` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `83` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `60` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `63` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `54` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `83` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `58` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `53` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `38` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `37` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `64` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `65` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `51` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `54` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `50` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `49` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `41` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `42` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `35` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `45` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `38` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `63` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `61` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-cpu_refit-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"cpu_refit"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-cpu_refit-V2.json"` |
| `score.sha256` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1748` |
| `counts.false_positives` | `6036` |
| `counts.false_negatives` | `889` |
| `metrics.recall` | `0.6628744785741373` |
| `metrics.precision` | `0.22456320657759507` |
| `metrics.frauds_per_1000_reviews` | `224.56320657759505` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6628744785741373` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.627211931088966` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.605185925795854` |
| `metrics.fraud_value_exposed` | `159016.392` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `57` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `96` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `67` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `77` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `70` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `103` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `62` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `49` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `80` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `35` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `76` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `70` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `60` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `43` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `61` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `57` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `42` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `75` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `70` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `93` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `55` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1a9a3415f7d97343ba28ce4f768638ec17e05473242336819881a8132f2d99af"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `649` |
| `counts.false_positives` | `135` |
| `counts.false_negatives` | `1988` |
| `metrics.recall` | `0.24611300720515739` |
| `metrics.precision` | `0.8278061224489796` |
| `metrics.frauds_per_1000_reviews` | `827.8061224489796` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8278061224489796` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.429855161632343` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.17126994105839846` |
| `metrics.fraud_value_exposed` | `333781.576` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1a9a3415f7d97343ba28ce4f768638ec17e05473242336819881a8132f2d99af"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1049` |
| `counts.false_positives` | `519` |
| `counts.false_negatives` | `1588` |
| `metrics.recall` | `0.39780053090633294` |
| `metrics.precision` | `0.6690051020408163` |
| `metrics.frauds_per_1000_reviews` | `669.0051020408164` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6690051020408163` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.743388339408572` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.28644667970812515` |
| `metrics.fraud_value_exposed` | `287392.679` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1a9a3415f7d97343ba28ce4f768638ec17e05473242336819881a8132f2d99af"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1473` |
| `counts.false_positives` | `2419` |
| `counts.false_negatives` | `1164` |
| `metrics.recall` | `0.558589306029579` |
| `metrics.precision` | `0.3784686536485098` |
| `metrics.frauds_per_1000_reviews` | `378.46865364850976` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.558589306029579` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.16920271681241` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.467518707118487` |
| `metrics.fraud_value_exposed` | `214463.616` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `49` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `83` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `58` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `66` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `55` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `86` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `58` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `52` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `38` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `36` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `61` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `25` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `65` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `54` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `57` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `42` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `39` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `47` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `39` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `48` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `40` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `66` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `62` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `47` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-Cat-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1a9a3415f7d97343ba28ce4f768638ec17e05473242336819881a8132f2d99af"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1769` |
| `counts.false_positives` | `6015` |
| `counts.false_negatives` | `868` |
| `metrics.recall` | `0.670838073568449` |
| `metrics.precision` | `0.22726104830421376` |
| `metrics.frauds_per_1000_reviews` | `227.26104830421377` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.670838073568449` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.706829465730195` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6192892691827161` |
| `metrics.fraud_value_exposed` | `153336.091` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `52` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `97` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `66` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `79` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `68` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `104` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `62` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `65` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `53` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `77` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `33` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `76` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `76` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `69` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `61` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `45` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `60` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `49` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `55` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `46` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `76` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `73` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `96` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `56` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"522d25b74dc8fb4e16391c1e184ce4284e35521e37c1bffa180e4a483dc965ba"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `658` |
| `counts.false_positives` | `126` |
| `counts.false_negatives` | `1979` |
| `metrics.recall` | `0.24952597648843383` |
| `metrics.precision` | `0.8392857142857143` |
| `metrics.frauds_per_1000_reviews` | `839.2857142857143` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8392857142857143` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.768635895769002` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.18131459702810035` |
| `metrics.fraud_value_exposed` | `329735.963` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"522d25b74dc8fb4e16391c1e184ce4284e35521e37c1bffa180e4a483dc965ba"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1022` |
| `counts.false_positives` | `546` |
| `counts.false_negatives` | `1615` |
| `metrics.recall` | `0.3875616230565036` |
| `metrics.precision` | `0.6517857142857143` |
| `metrics.frauds_per_1000_reviews` | `651.7857142857143` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6517857142857143` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.235217238203585` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.28125345581881606` |
| `metrics.fraud_value_exposed` | `289484.316` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `19` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"522d25b74dc8fb4e16391c1e184ce4284e35521e37c1bffa180e4a483dc965ba"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1464` |
| `counts.false_positives` | `2428` |
| `counts.false_negatives` | `1173` |
| `metrics.recall` | `0.5551763367463026` |
| `metrics.precision` | `0.3761562178828366` |
| `metrics.frauds_per_1000_reviews` | `376.1562178828366` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5551763367463026` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.100959115691357` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4630498427158318` |
| `metrics.fraud_value_exposed` | `216263.508` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `49` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `36` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `79` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `57` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `66` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `57` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `82` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `57` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `53` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `36` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `36` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `64` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `67` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `53` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `51` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `47` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `40` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `46` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `37` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `47` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `39` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `62` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `62` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `88` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-Cat-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"522d25b74dc8fb4e16391c1e184ce4284e35521e37c1bffa180e4a483dc965ba"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1747` |
| `counts.false_positives` | `6037` |
| `counts.false_negatives` | `890` |
| `metrics.recall` | `0.6624952597648843` |
| `metrics.precision` | `0.22443473792394655` |
| `metrics.frauds_per_1000_reviews` | `224.43473792394656` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6624952597648843` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.623420619915574` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6008711074265155` |
| `metrics.fraud_value_exposed` | `160754.24` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `54` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `97` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `67` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `79` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `65` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `104` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `63` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `50` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `80` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `32` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `75` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `72` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `67` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `62` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `45` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `58` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `46` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `54` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `46` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `75` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `70` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `93` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `55` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"f6767e8aec9b771cc7f0f2dbf1d2ffb403eb9200bf9e1c01b8de85f906002b9c"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `642` |
| `counts.false_positives` | `142` |
| `counts.false_negatives` | `1995` |
| `metrics.recall` | `0.2434584755403868` |
| `metrics.precision` | `0.8188775510204082` |
| `metrics.frauds_per_1000_reviews` | `818.8775510204082` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8188775510204082` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.16635903508161` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16806461463896544` |
| `metrics.fraud_value_exposed` | `335072.562` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"f6767e8aec9b771cc7f0f2dbf1d2ffb403eb9200bf9e1c01b8de85f906002b9c"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1027` |
| `counts.false_positives` | `541` |
| `counts.false_negatives` | `1610` |
| `metrics.recall` | `0.3894577171027683` |
| `metrics.precision` | `0.6549744897959183` |
| `metrics.frauds_per_1000_reviews` | `654.9744897959183` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6549744897959183` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.329322997685992` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2796918266944684` |
| `metrics.fraud_value_exposed` | `290113.282` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `53` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `49` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `17` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"f6767e8aec9b771cc7f0f2dbf1d2ffb403eb9200bf9e1c01b8de85f906002b9c"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1473` |
| `counts.false_positives` | `2419` |
| `counts.false_negatives` | `1164` |
| `metrics.recall` | `0.558589306029579` |
| `metrics.precision` | `0.3784686536485098` |
| `metrics.frauds_per_1000_reviews` | `378.46865364850976` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.558589306029579` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.16920271681241` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.46970729346839957` |
| `metrics.fraud_value_exposed` | `213582.135` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `47` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `84` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `59` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `64` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `57` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `83` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `58` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `55` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `38` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `64` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `25` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `54` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `57` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `46` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `37` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `45` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `39` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `47` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `39` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `65` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `61` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `50` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-Cat-u-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-Cat-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"f6767e8aec9b771cc7f0f2dbf1d2ffb403eb9200bf9e1c01b8de85f906002b9c"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1756` |
| `counts.false_positives` | `6028` |
| `counts.false_negatives` | `881` |
| `metrics.recall` | `0.6659082290481608` |
| `metrics.precision` | `0.22559095580678315` |
| `metrics.frauds_per_1000_reviews` | `225.59095580678314` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6659082290481608` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.6575424204761005` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6262350435389349` |
| `metrics.fraud_value_exposed` | `150538.592` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `52` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `96` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `67` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `78` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `68` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `105` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `63` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `55` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `80` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `29` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `76` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `70` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `67` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `59` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `44` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `58` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `48` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `55` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `44` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `75` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `72` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `96` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `58` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"b6a3d5673c00f4a686f335a9c1f1352e0814f1d1892e32c38299c67243f03d27"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `554` |
| `counts.false_positives` | `230` |
| `counts.false_negatives` | `2083` |
| `metrics.recall` | `0.21008722032612817` |
| `metrics.precision` | `0.7066326530612245` |
| `metrics.frauds_per_1000_reviews` | `706.6326530612245` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7066326530612245` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `20.85383630130095` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.12879396455765357` |
| `metrics.fraud_value_exposed` | `350889.316` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `9` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"b6a3d5673c00f4a686f335a9c1f1352e0814f1d1892e32c38299c67243f03d27"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `856` |
| `counts.false_positives` | `712` |
| `counts.false_negatives` | `1781` |
| `metrics.recall` | `0.32461130072051575` |
| `metrics.precision` | `0.5459183673469388` |
| `metrics.frauds_per_1000_reviews` | `545.9183673469388` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5459183673469388` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.110906023387738` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2544658378067778` |
| `metrics.fraud_value_exposed` | `300273.37` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

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
| `148` | `3060` | `86` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `13` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"b6a3d5673c00f4a686f335a9c1f1352e0814f1d1892e32c38299c67243f03d27"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1268` |
| `counts.false_positives` | `2624` |
| `counts.false_negatives` | `1369` |
| `metrics.recall` | `0.48084945013272656` |
| `metrics.precision` | `0.32579650565262075` |
| `metrics.frauds_per_1000_reviews` | `325.79650565262074` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.48084945013272656` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.614765135721749` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4219214512016261` |
| `metrics.fraud_value_exposed` | `232828.491` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `47` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `27` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `78` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `53` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `51` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `42` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `71` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `52` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `47` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `35` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `30` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `51` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `19` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `50` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `50` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `54` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `44` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `46` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `31` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `51` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `34` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `33` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `27` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `24` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `52` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `49` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `78` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `42` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"b6a3d5673c00f4a686f335a9c1f1352e0814f1d1892e32c38299c67243f03d27"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1542` |
| `counts.false_positives` | `6242` |
| `counts.false_negatives` | `1095` |
| `metrics.recall` | `0.5847554038680318` |
| `metrics.precision` | `0.19809866392600206` |
| `metrics.frauds_per_1000_reviews` | `198.09866392600205` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5847554038680318` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.846201829370243` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5587750071925723` |
| `metrics.fraud_value_exposed` | `177708.98` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `34` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `93` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `64` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `61` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `58` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `88` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `54` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `52` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `44` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `66` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `26` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `59` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `66` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `67` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `50` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `58` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `37` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `65` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `40` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `40` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `34` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `31` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `60` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `62` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `85` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `52` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"308092a320b1b8096cb71086e7aa0db5a9a5fc9d9963638d9d4ffd71e1ada8a8"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `535` |
| `counts.false_positives` | `249` |
| `counts.false_negatives` | `2102` |
| `metrics.recall` | `0.20288206295032232` |
| `metrics.precision` | `0.6823979591836735` |
| `metrics.frauds_per_1000_reviews` | `682.3979591836735` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.6823979591836735` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `20.138632529234673` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.15844172312232152` |
| `metrics.fraud_value_exposed` | `338948.304` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `8` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `8` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"308092a320b1b8096cb71086e7aa0db5a9a5fc9d9963638d9d4ffd71e1ada8a8"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `854` |
| `counts.false_positives` | `714` |
| `counts.false_negatives` | `1783` |
| `metrics.recall` | `0.32385286310200984` |
| `metrics.precision` | `0.5446428571428571` |
| `metrics.frauds_per_1000_reviews` | `544.6428571428571` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5446428571428571` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.073263719594777` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2698227803226398` |
| `metrics.fraud_value_exposed` | `294088.166` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `53` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `9` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `22` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `12` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `54` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"308092a320b1b8096cb71086e7aa0db5a9a5fc9d9963638d9d4ffd71e1ada8a8"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1235` |
| `counts.false_positives` | `2657` |
| `counts.false_negatives` | `1402` |
| `metrics.recall` | `0.4683352294273796` |
| `metrics.precision` | `0.31731757451181913` |
| `metrics.frauds_per_1000_reviews` | `317.3175745118191` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.4683352294273796` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.364538598277887` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4100140061620996` |
| `metrics.fraud_value_exposed` | `237624.366` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `50` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `29` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `81` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `53` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `48` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `42` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `62` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `50` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `39` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `37` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `50` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `13` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `45` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `59` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `56` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `41` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `43` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `32` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `44` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `28` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `36` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `29` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `19` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `55` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `42` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `75` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `46` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"308092a320b1b8096cb71086e7aa0db5a9a5fc9d9963638d9d4ffd71e1ada8a8"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1514` |
| `counts.false_positives` | `6270` |
| `counts.false_negatives` | `1123` |
| `metrics.recall` | `0.5741372772089496` |
| `metrics.precision` | `0.19450154162384378` |
| `metrics.frauds_per_1000_reviews` | `194.5015416238438` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5741372772089496` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.7400451165152715` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5292964996432987` |
| `metrics.fraud_value_exposed` | `189581.824` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `56` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `38` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `99` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `59` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `57` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `55` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `86` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `56` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `48` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `45` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `34` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `63` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `20` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `56` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `64` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `71` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `48` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `59` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `34` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `56` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `42` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `44` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `37` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `29` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `60` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `54` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `57` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0df888edd13c14014f9978bed7bef834b41d8e81614fcbb5c9bf71a63796c19b"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `569` |
| `counts.false_positives` | `215` |
| `counts.false_negatives` | `2068` |
| `metrics.recall` | `0.21577550246492225` |
| `metrics.precision` | `0.7257653061224489` |
| `metrics.frauds_per_1000_reviews` | `725.765306122449` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7257653061224489` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `21.418470858195384` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16127933676404474` |
| `metrics.fraud_value_exposed` | `337805.419` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0df888edd13c14014f9978bed7bef834b41d8e81614fcbb5c9bf71a63796c19b"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `885` |
| `counts.false_positives` | `683` |
| `counts.false_negatives` | `1752` |
| `metrics.recall` | `0.33560864618885095` |
| `metrics.precision` | `0.5644132653061225` |
| `metrics.frauds_per_1000_reviews` | `564.4132653061224` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5644132653061225` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.65671942838569` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2753257548233287` |
| `metrics.fraud_value_exposed` | `291871.773` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `13` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `15` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0df888edd13c14014f9978bed7bef834b41d8e81614fcbb5c9bf71a63796c19b"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1315` |
| `counts.false_positives` | `2577` |
| `counts.false_negatives` | `1322` |
| `metrics.recall` | `0.4986727341676147` |
| `metrics.precision` | `0.33787255909558067` |
| `metrics.frauds_per_1000_reviews` | `337.8725590955807` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.4986727341676147` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.971148386020584` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4294041098997037` |
| `metrics.fraud_value_exposed` | `229814.755` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `50` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `32` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `83` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `57` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `47` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `48` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `74` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `52` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `45` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `35` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `30` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `61` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `19` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `51` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `44` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `42` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `34` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `50` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `29` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `35` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `31` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `22` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `56` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `53` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `81` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-b-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0df888edd13c14014f9978bed7bef834b41d8e81614fcbb5c9bf71a63796c19b"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1591` |
| `counts.false_positives` | `6193` |
| `counts.false_negatives` | `1046` |
| `metrics.recall` | `0.6033371255214258` |
| `metrics.precision` | `0.20439362795477903` |
| `metrics.frauds_per_1000_reviews` | `204.39362795477902` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6033371255214258` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.031976076866444` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5459245702191646` |
| `metrics.fraud_value_exposed` | `182884.657` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `57` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `38` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `93` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `70` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `58` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `91` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `59` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `54` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `49` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `36` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `70` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `25` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `67` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `65` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `49` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `51` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `43` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `59` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `37` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `41` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `40` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `34` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `62` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `64` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `89` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `64` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"fe96eb55d02b6115e79e7b4a447e4b996341d59031a2cfed12cfc8b4288e7332"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `572` |
| `counts.false_positives` | `212` |
| `counts.false_negatives` | `2065` |
| `metrics.recall` | `0.2169131588926811` |
| `metrics.precision` | `0.7295918367346939` |
| `metrics.frauds_per_1000_reviews` | `729.5918367346939` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7295918367346939` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `21.53139776957427` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.14640743913530588` |
| `metrics.fraud_value_exposed` | `343795.265` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `7` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"fe96eb55d02b6115e79e7b4a447e4b996341d59031a2cfed12cfc8b4288e7332"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `908` |
| `counts.false_positives` | `660` |
| `counts.false_negatives` | `1729` |
| `metrics.recall` | `0.34433067880166857` |
| `metrics.precision` | `0.5790816326530612` |
| `metrics.frauds_per_1000_reviews` | `579.0816326530612` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5790816326530612` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `17.08960592200475` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.28063532513003864` |
| `metrics.fraud_value_exposed` | `289733.276` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

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
| `148` | `3060` | `86` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"fe96eb55d02b6115e79e7b4a447e4b996341d59031a2cfed12cfc8b4288e7332"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1246` |
| `counts.false_positives` | `2646` |
| `counts.false_negatives` | `1391` |
| `metrics.recall` | `0.4725066363291619` |
| `metrics.precision` | `0.32014388489208634` |
| `metrics.frauds_per_1000_reviews` | `320.1438848920863` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.4725066363291619` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.447947444092508` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.43032236270206575` |
| `metrics.fraud_value_exposed` | `229444.917` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `46` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `73` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `55` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `46` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `41` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `68` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `49` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `36` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `26` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `51` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `14` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `50` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `53` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `44` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `45` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `42` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `32` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `44` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `35` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `33` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `28` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `29` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `51` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `55` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `79` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `35` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-FTT-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"fe96eb55d02b6115e79e7b4a447e4b996341d59031a2cfed12cfc8b4288e7332"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1538` |
| `counts.false_positives` | `6246` |
| `counts.false_negatives` | `1099` |
| `metrics.recall` | `0.5832385286310201` |
| `metrics.precision` | `0.19758478931140802` |
| `metrics.frauds_per_1000_reviews` | `197.584789311408` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5832385286310201` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.831036584676676` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5955804539647517` |
| `metrics.fraud_value_exposed` | `162885.118` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `37` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `88` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `59` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `61` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `54` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `90` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `56` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `57` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `47` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `33` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `64` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `23` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `67` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `60` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `60` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `40` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `53` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `42` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `44` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `38` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `39` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `61` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `67` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `85` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b10d5e80dd3fa1e3baa334e439817292ecba6dc9abc25e5192235c87eb53bc20"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `566` |
| `counts.false_positives` | `218` |
| `counts.false_negatives` | `2071` |
| `metrics.recall` | `0.21463784603716343` |
| `metrics.precision` | `0.7219387755102041` |
| `metrics.frauds_per_1000_reviews` | `721.9387755102041` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7219387755102041` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `21.305543946816496` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.17254439862215254` |
| `metrics.fraud_value_exposed` | `333268.272` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b10d5e80dd3fa1e3baa334e439817292ecba6dc9abc25e5192235c87eb53bc20"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `906` |
| `counts.false_positives` | `662` |
| `counts.false_negatives` | `1731` |
| `metrics.recall` | `0.3435722411831627` |
| `metrics.precision` | `0.5778061224489796` |
| `metrics.frauds_per_1000_reviews` | `577.8061224489796` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5778061224489796` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `17.05196361821179` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2903238309610797` |
| `metrics.fraud_value_exposed` | `285831.107` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `13` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b10d5e80dd3fa1e3baa334e439817292ecba6dc9abc25e5192235c87eb53bc20"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1257` |
| `counts.false_positives` | `2635` |
| `counts.false_negatives` | `1380` |
| `metrics.recall` | `0.47667804323094426` |
| `metrics.precision` | `0.32297019527235354` |
| `metrics.frauds_per_1000_reviews` | `322.9701952723535` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.47667804323094426` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.531356289907128` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4256618842413033` |
| `metrics.fraud_value_exposed` | `231321.984` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `47` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `74` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `56` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `47` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `43` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `71` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `52` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `47` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `31` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `27` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `48` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `21` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `47` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `56` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `56` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `45` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `38` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `30` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `44` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `31` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `36` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `32` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `28` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `49` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `57` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `77` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `36` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-FTT-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b10d5e80dd3fa1e3baa334e439817292ecba6dc9abc25e5192235c87eb53bc20"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1566` |
| `counts.false_positives` | `6218` |
| `counts.false_negatives` | `1071` |
| `metrics.recall` | `0.5938566552901023` |
| `metrics.precision` | `0.2011819116135663` |
| `metrics.frauds_per_1000_reviews` | `201.18191161356629` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5938566552901023` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.937193297531648` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5729984187911095` |
| `metrics.fraud_value_exposed` | `171980.32` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `52` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `85` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `65` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `60` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `57` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `89` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `61` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `55` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `50` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `31` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `60` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `34` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `61` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `69` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `68` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `49` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `51` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `39` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `59` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `34` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `49` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `37` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `37` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `62` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `67` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `92` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `51` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"4521c4c7746e29af794a88c36ed376e851fcc4a2c3d3084d38aad486e32d6103"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `557` |
| `counts.false_positives` | `227` |
| `counts.false_negatives` | `2080` |
| `metrics.recall` | `0.211224876753887` |
| `metrics.precision` | `0.7104591836734694` |
| `metrics.frauds_per_1000_reviews` | `710.4591836734694` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7104591836734694` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `20.96676321267984` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.12294078138174476` |
| `metrics.fraud_value_exposed` | `353246.76` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `9` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"4521c4c7746e29af794a88c36ed376e851fcc4a2c3d3084d38aad486e32d6103"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `882` |
| `counts.false_positives` | `686` |
| `counts.false_negatives` | `1755` |
| `metrics.recall` | `0.33447098976109213` |
| `metrics.precision` | `0.5625` |
| `metrics.frauds_per_1000_reviews` | `562.5` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5625` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.600255972696246` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.21884619093995403` |
| `metrics.fraud_value_exposed` | `314619.636` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `52` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `12` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `22` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `22` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"4521c4c7746e29af794a88c36ed376e851fcc4a2c3d3084d38aad486e32d6103"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1323` |
| `counts.false_positives` | `2569` |
| `counts.false_negatives` | `1314` |
| `metrics.recall` | `0.5017064846416383` |
| `metrics.precision` | `0.33992805755395683` |
| `metrics.frauds_per_1000_reviews` | `339.9280575539568` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5017064846416383` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `10.031809364794853` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.43864999284293255` |
| `metrics.fraud_value_exposed` | `226090.858` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `47` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `80` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `54` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `56` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `44` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `74` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `56` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `52` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `35` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `51` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `22` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `56` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `51` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `44` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `42` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `28` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `44` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `33` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `37` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `30` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `61` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `58` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `75` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `40` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-FTT-u-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"FTT-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-FTT-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"4521c4c7746e29af794a88c36ed376e851fcc4a2c3d3084d38aad486e32d6103"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1619` |
| `counts.false_positives` | `6165` |
| `counts.false_negatives` | `1018` |
| `metrics.recall` | `0.6139552521805082` |
| `metrics.precision` | `0.2079907502569373` |
| `metrics.frauds_per_1000_reviews` | `207.9907502569373` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6139552521805082` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.138132789721417` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5682876675548555` |
| `metrics.fraud_value_exposed` | `173877.635` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `44` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `94` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `68` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `64` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `95` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `61` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `44` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `67` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `68` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `65` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `49` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `51` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `37` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `50` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `42` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `51` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `37` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `68` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `71` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `85` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `56` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"7fe3d04d2c96421bcfd94b61164a66fed588a0244f45ad0670af89acc6292e77"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `650` |
| `counts.false_positives` | `134` |
| `counts.false_negatives` | `1987` |
| `metrics.recall` | `0.24649222601441031` |
| `metrics.precision` | `0.8290816326530612` |
| `metrics.frauds_per_1000_reviews` | `829.0816326530612` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8290816326530612` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.467497465425307` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.17419375433561723` |
| `metrics.fraud_value_exposed` | `332603.973` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `9` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"7fe3d04d2c96421bcfd94b61164a66fed588a0244f45ad0670af89acc6292e77"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1070` |
| `counts.false_positives` | `498` |
| `counts.false_negatives` | `1567` |
| `metrics.recall` | `0.40576412590064465` |
| `metrics.precision` | `0.6823979591836735` |
| `metrics.frauds_per_1000_reviews` | `682.3979591836735` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6823979591836735` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `20.138632529234673` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.3172837340329439` |
| `metrics.fraud_value_exposed` | `274972.663` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

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
| `148` | `3060` | `86` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"7fe3d04d2c96421bcfd94b61164a66fed588a0244f45ad0670af89acc6292e77"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1580` |
| `counts.false_positives` | `2312` |
| `counts.false_negatives` | `1057` |
| `metrics.recall` | `0.5991657186196435` |
| `metrics.precision` | `0.40596094552929085` |
| `metrics.frauds_per_1000_reviews` | `405.9609455292908` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5991657186196435` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.980543307918268` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.559595058255689` |
| `metrics.fraud_value_exposed` | `177378.694` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `93` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `61` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `68` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `59` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `96` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `60` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `55` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `45` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `41` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `66` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `30` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `65` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `62` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `65` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `49` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `43` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `38` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `46` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `43` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `45` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `44` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `72` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `68` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `91` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `52` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"7fe3d04d2c96421bcfd94b61164a66fed588a0244f45ad0670af89acc6292e77"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1858` |
| `counts.false_positives` | `5926` |
| `counts.false_negatives` | `779` |
| `metrics.recall` | `0.7045885475919605` |
| `metrics.precision` | `0.23869475847893115` |
| `metrics.frauds_per_1000_reviews` | `238.69475847893113` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.7045885475919605` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `7.04425616016207` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6911164698548419` |
| `metrics.fraud_value_exposed` | `124406.772` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `63` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `107` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `73` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `79` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `77` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `110` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `67` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `65` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `52` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `49` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `78` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `35` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `80` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `81` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `73` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `57` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `61` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `44` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `66` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `46` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `49` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `47` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `38` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `80` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `80` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `97` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `64` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"8a958e7b1aa31e3bfaf9c1a64cae109c10f54c093cbc56c08e4c9cb65b35ada0"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `645` |
| `counts.false_positives` | `139` |
| `counts.false_negatives` | `1992` |
| `metrics.recall` | `0.2445961319681456` |
| `metrics.precision` | `0.8227040816326531` |
| `metrics.frauds_per_1000_reviews` | `822.704081632653` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8227040816326531` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.279285946460494` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16042254446565915` |
| `metrics.fraud_value_exposed` | `338150.503` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `9` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"8a958e7b1aa31e3bfaf9c1a64cae109c10f54c093cbc56c08e4c9cb65b35ada0"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1067` |
| `counts.false_positives` | `501` |
| `counts.false_negatives` | `1570` |
| `metrics.recall` | `0.40462646947288583` |
| `metrics.precision` | `0.6804846938775511` |
| `metrics.frauds_per_1000_reviews` | `680.484693877551` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6804846938775511` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `20.08216907354523` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.32174688291163706` |
| `metrics.fraud_value_exposed` | `273175.073` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `54` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"8a958e7b1aa31e3bfaf9c1a64cae109c10f54c093cbc56c08e4c9cb65b35ada0"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1542` |
| `counts.false_positives` | `2350` |
| `counts.false_negatives` | `1095` |
| `metrics.recall` | `0.5847554038680318` |
| `metrics.precision` | `0.3961973278520041` |
| `metrics.frauds_per_1000_reviews` | `396.1973278520041` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5847554038680318` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.692403658740487` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.5246524849702824` |
| `metrics.fraud_value_exposed` | `191452.26` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `90` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `65` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `68` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `60` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `94` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `59` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `55` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `45` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `36` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `66` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `29` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `61` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `62` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `48` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `42` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `38` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `47` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `44` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `43` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `39` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `65` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `65` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `50` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"8a958e7b1aa31e3bfaf9c1a64cae109c10f54c093cbc56c08e4c9cb65b35ada0"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1823` |
| `counts.false_positives` | `5961` |
| `counts.false_negatives` | `814` |
| `metrics.recall` | `0.6913158892681077` |
| `metrics.precision` | `0.2341983556012333` |
| `metrics.frauds_per_1000_reviews` | `234.1983556012333` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6913158892681077` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.911560269093355` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6636321984951119` |
| `metrics.fraud_value_exposed` | `135476.412` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `57` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `103` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `71` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `79` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `80` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `109` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `66` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `65` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `55` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `73` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `34` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `79` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `75` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `72` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `52` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `58` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `47` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `65` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `49` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `47` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `79` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `78` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `94` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `62` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"96df74dd133fc30c2314dbe7094030d2f0035cc49604fe02e69462064bc44eaf"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `646` |
| `counts.false_positives` | `138` |
| `counts.false_negatives` | `1991` |
| `metrics.recall` | `0.24497535077739857` |
| `metrics.precision` | `0.8239795918367347` |
| `metrics.frauds_per_1000_reviews` | `823.9795918367347` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8239795918367347` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.316928250253458` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16781061893900587` |
| `metrics.fraud_value_exposed` | `335174.862` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"96df74dd133fc30c2314dbe7094030d2f0035cc49604fe02e69462064bc44eaf"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1061` |
| `counts.false_positives` | `507` |
| `counts.false_negatives` | `1576` |
| `metrics.recall` | `0.4023511566173682` |
| `metrics.precision` | `0.6766581632653061` |
| `metrics.frauds_per_1000_reviews` | `676.6581632653061` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6766581632653061` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.969242162166346` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.3037965151909142` |
| `metrics.fraud_value_exposed` | `280404.812` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `54` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"96df74dd133fc30c2314dbe7094030d2f0035cc49604fe02e69462064bc44eaf"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1572` |
| `counts.false_positives` | `2320` |
| `counts.false_negatives` | `1065` |
| `metrics.recall` | `0.59613196814562` |
| `metrics.precision` | `0.4039054470709147` |
| `metrics.frauds_per_1000_reviews` | `403.9054470709147` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.59613196814562` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.919882329143999` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.5489451973216867` |
| `metrics.fraud_value_exposed` | `181668.061` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `92` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `67` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `62` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `90` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `59` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `59` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `45` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `39` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `63` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `31` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `66` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `67` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `61` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `49` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `45` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `39` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `52` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `45` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `43` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `40` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `70` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `65` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `89` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `50` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-b-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"96df74dd133fc30c2314dbe7094030d2f0035cc49604fe02e69462064bc44eaf"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1840` |
| `counts.false_positives` | `5944` |
| `counts.false_negatives` | `797` |
| `metrics.recall` | `0.6977626090254077` |
| `metrics.precision` | `0.23638232271325796` |
| `metrics.frauds_per_1000_reviews` | `236.38232271325796` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6977626090254077` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.976012559041017` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6751708879593337` |
| `metrics.fraud_value_exposed` | `130829.058` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `60` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `39` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `107` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `73` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `75` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `82` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `108` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `66` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `66` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `52` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `47` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `75` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `36` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `79` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `79` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `71` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `55` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `57` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `65` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `52` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `48` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `43` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `79` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `76` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `96` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `61` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"d66fcf663107e4e2b616e1795573e399c784a323ec8792f802ae4e1aa728bc46"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `640` |
| `counts.false_positives` | `144` |
| `counts.false_negatives` | `1997` |
| `metrics.recall` | `0.2427000379218809` |
| `metrics.precision` | `0.8163265306122449` |
| `metrics.frauds_per_1000_reviews` | `816.3265306122449` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8163265306122449` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.091074427495684` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16066057046530455` |
| `metrics.fraud_value_exposed` | `338054.635` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `8` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"d66fcf663107e4e2b616e1795573e399c784a323ec8792f802ae4e1aa728bc46"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1062` |
| `counts.false_positives` | `506` |
| `counts.false_negatives` | `1575` |
| `metrics.recall` | `0.40273037542662116` |
| `metrics.precision` | `0.6772959183673469` |
| `metrics.frauds_per_1000_reviews` | `677.295918367347` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6772959183673469` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.988063314062828` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.29359388035125117` |
| `metrics.fraud_value_exposed` | `284514.053` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

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
| `148` | `3060` | `86` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"d66fcf663107e4e2b616e1795573e399c784a323ec8792f802ae4e1aa728bc46"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1584` |
| `counts.false_positives` | `2308` |
| `counts.false_negatives` | `1053` |
| `metrics.recall` | `0.6006825938566553` |
| `metrics.precision` | `0.40698869475847893` |
| `metrics.frauds_per_1000_reviews` | `406.9886947584789` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6006825938566553` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `12.010873797305402` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.5485647351019505` |
| `metrics.fraud_value_exposed` | `181821.297` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `56` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `93` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `61` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `69` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `60` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `89` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `60` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `63` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `41` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `43` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `63` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `25` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `66` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `63` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `46` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `53` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `40` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `48` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `43` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `40` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `70` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `69` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `91` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `52` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-LGB-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"d66fcf663107e4e2b616e1795573e399c784a323ec8792f802ae4e1aa728bc46"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1857` |
| `counts.false_positives` | `5927` |
| `counts.false_negatives` | `780` |
| `metrics.recall` | `0.7042093287827076` |
| `metrics.precision` | `0.23856628982528263` |
| `metrics.frauds_per_1000_reviews` | `238.56628982528264` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.7042093287827076` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `7.0404648489886785` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6834903718647012` |
| `metrics.fraud_value_exposed` | `127478.28` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `59` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `108` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `74` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `77` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `75` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `112` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `72` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `68` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `52` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `49` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `75` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `31` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `80` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `80` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `71` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `52` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `63` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `67` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `54` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `46` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `81` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `79` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `97` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `60` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"29163846b44103353e020e881b6a9e99e27be338e8b864a7dd3888908ffdeecd"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `637` |
| `counts.false_positives` | `147` |
| `counts.false_negatives` | `2000` |
| `metrics.recall` | `0.2415623814941221` |
| `metrics.precision` | `0.8125` |
| `metrics.frauds_per_1000_reviews` | `812.5` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8125` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `23.9781475161168` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.15880376010169203` |
| `metrics.fraud_value_exposed` | `338802.489` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `8` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"29163846b44103353e020e881b6a9e99e27be338e8b864a7dd3888908ffdeecd"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1058` |
| `counts.false_positives` | `510` |
| `counts.false_negatives` | `1579` |
| `metrics.recall` | `0.4012135001896094` |
| `metrics.precision` | `0.6747448979591837` |
| `metrics.frauds_per_1000_reviews` | `674.7448979591836` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6747448979591837` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `19.912778706476903` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.29237851713407326` |
| `metrics.fraud_value_exposed` | `285003.556` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"29163846b44103353e020e881b6a9e99e27be338e8b864a7dd3888908ffdeecd"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1569` |
| `counts.false_positives` | `2323` |
| `counts.false_negatives` | `1068` |
| `metrics.recall` | `0.5949943117178612` |
| `metrics.precision` | `0.40313463514902365` |
| `metrics.frauds_per_1000_reviews` | `403.13463514902367` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5949943117178612` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.897134462103647` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.5346858067232657` |
| `metrics.fraud_value_exposed` | `187411.212` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `90` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `69` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `63` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `90` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `58` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `62` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `39` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `38` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `65` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `25` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `65` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `66` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `61` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `50` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `39` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `48` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `46` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `43` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `40` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `35` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `70` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `68` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `89` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `50` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-LGB-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"29163846b44103353e020e881b6a9e99e27be338e8b864a7dd3888908ffdeecd"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1850` |
| `counts.false_positives` | `5934` |
| `counts.false_negatives` | `787` |
| `metrics.recall` | `0.701554797117937` |
| `metrics.precision` | `0.23766700924974307` |
| `metrics.frauds_per_1000_reviews` | `237.66700924974307` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.701554797117937` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `7.013925670774936` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6781449491835296` |
| `metrics.fraud_value_exposed` | `129631.217` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `58` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `105` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `73` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `77` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `74` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `111` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `66` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `70` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `50` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `48` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `79` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `33` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `79` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `75` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `71` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `54` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `62` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `67` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `48` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `53` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `48` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `46` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `80` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `77` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `98` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `62` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0c6de8e326b95199ea00e2174e0e50d61a916d7fbaa08660717a280d10e3501f"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `643` |
| `counts.false_positives` | `141` |
| `counts.false_negatives` | `1994` |
| `metrics.recall` | `0.24383769434963976` |
| `metrics.precision` | `0.8201530612244898` |
| `metrics.frauds_per_1000_reviews` | `820.1530612244898` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.8201530612244898` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `24.204001338874573` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16306907289662687` |
| `metrics.fraud_value_exposed` | `337084.58` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `8` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0c6de8e326b95199ea00e2174e0e50d61a916d7fbaa08660717a280d10e3501f"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `1076` |
| `counts.false_positives` | `492` |
| `counts.false_negatives` | `1561` |
| `metrics.recall` | `0.40803943875616233` |
| `metrics.precision` | `0.6862244897959183` |
| `metrics.frauds_per_1000_reviews` | `686.2244897959183` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.6862244897959183` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `20.251559440613562` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.3033546297099729` |
| `metrics.fraud_value_exposed` | `280582.787` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `43` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `22` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `42` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `40` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `56` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0c6de8e326b95199ea00e2174e0e50d61a916d7fbaa08660717a280d10e3501f"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1572` |
| `counts.false_positives` | `2320` |
| `counts.false_negatives` | `1065` |
| `metrics.recall` | `0.59613196814562` |
| `metrics.precision` | `0.4039054470709147` |
| `metrics.frauds_per_1000_reviews` | `403.9054470709147` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.59613196814562` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `11.919882329143999` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.5389709103268454` |
| `metrics.fraud_value_exposed` | `185685.332` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `94` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `63` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `66` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `63` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `90` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `58` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `62` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `40` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `39` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `66` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `64` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `61` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `54` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `40` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `50` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `44` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `41` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `39` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `69` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `68` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `90` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `51` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-LGB-u-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"LGB-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-LGB-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"0c6de8e326b95199ea00e2174e0e50d61a916d7fbaa08660717a280d10e3501f"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1848` |
| `counts.false_positives` | `5936` |
| `counts.false_negatives` | `789` |
| `metrics.recall` | `0.7007963594994312` |
| `metrics.precision` | `0.23741007194244604` |
| `metrics.frauds_per_1000_reviews` | `237.41007194244605` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.7007963594994312` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `7.0063430484281515` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6754728026916413` |
| `metrics.fraud_value_exposed` | `130707.458` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `59` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `39` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `108` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `73` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `76` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `76` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `109` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `66` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `71` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `49` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `49` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `80` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `30` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `79` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `80` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `70` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `53` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `62` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `68` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `48` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `52` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `46` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `81` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `78` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `98` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `60` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1bd3f73ace57447940baa252512e545a01f66cf21885e806a83d12e4df000d4a"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `530` |
| `counts.false_positives` | `254` |
| `counts.false_negatives` | `2107` |
| `metrics.recall` | `0.20098596890405765` |
| `metrics.precision` | `0.6760204081632653` |
| `metrics.frauds_per_1000_reviews` | `676.0204081632653` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.6760204081632653` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `19.950421010269864` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.15348801742635945` |
| `metrics.fraud_value_exposed` | `340943.472` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `9` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1bd3f73ace57447940baa252512e545a01f66cf21885e806a83d12e4df000d4a"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `865` |
| `counts.false_positives` | `703` |
| `counts.false_negatives` | `1772` |
| `metrics.recall` | `0.3280242700037922` |
| `metrics.precision` | `0.5516581632653061` |
| `metrics.frauds_per_1000_reviews` | `551.6581632653061` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5516581632653061` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.28029639045607` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.28181926041398014` |
| `metrics.fraud_value_exposed` | `289256.431` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

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
| `148` | `3060` | `86` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `17` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `54` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1bd3f73ace57447940baa252512e545a01f66cf21885e806a83d12e4df000d4a"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1332` |
| `counts.false_positives` | `2560` |
| `counts.false_negatives` | `1305` |
| `metrics.recall` | `0.5051194539249146` |
| `metrics.precision` | `0.34224049331963` |
| `metrics.frauds_per_1000_reviews` | `342.24049331963` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5051194539249146` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `10.100052965915907` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4341307836620948` |
| `metrics.fraud_value_exposed` | `227911.027` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `47` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `31` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `77` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `52` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `46` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `53` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `80` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `51` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `37` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `34` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `29` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `53` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `28` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `55` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `64` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `46` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `45` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `34` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `50` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `43` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `39` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `34` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `32` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `39` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `57` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `76` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `47` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-b-same_seed_repeat-V2.json"` |
| `score.sha256` | `"1bd3f73ace57447940baa252512e545a01f66cf21885e806a83d12e4df000d4a"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1611` |
| `counts.false_positives` | `6173` |
| `counts.false_negatives` | `1026` |
| `metrics.recall` | `0.6109215017064846` |
| `metrics.precision` | `0.20696300102774923` |
| `metrics.frauds_per_1000_reviews` | `206.96300102774924` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.6109215017064846` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `6.107802300334281` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5694256949160965` |
| `metrics.fraud_value_exposed` | `173419.28` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `53` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `39` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `88` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `54` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `53` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `69` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `98` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `61` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `49` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `46` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `36` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `63` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `35` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `72` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `73` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `54` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `55` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `45` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `66` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `47` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `48` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `40` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `52` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `68` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `87` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `55` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b82d1a99f64a73cbea6c5566c17dfa2de884fac7a6dc4f978f593d1efa8c4d0e"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `521` |
| `counts.false_positives` | `263` |
| `counts.false_negatives` | `2116` |
| `metrics.recall` | `0.1975729996207812` |
| `metrics.precision` | `0.6645408163265306` |
| `metrics.frauds_per_1000_reviews` | `664.5408163265306` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.6645408163265306` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `19.611640276133205` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.14926463006045218` |
| `metrics.fraud_value_exposed` | `342644.495` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `13` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b82d1a99f64a73cbea6c5566c17dfa2de884fac7a6dc4f978f593d1efa8c4d0e"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `799` |
| `counts.false_positives` | `769` |
| `counts.false_negatives` | `1838` |
| `metrics.recall` | `0.3029958285930982` |
| `metrics.precision` | `0.5095663265306123` |
| `metrics.frauds_per_1000_reviews` | `509.56632653061223` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5095663265306123` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `15.038100365288322` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.25637926711410364` |
| `metrics.fraud_value_exposed` | `299502.712` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `16` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b82d1a99f64a73cbea6c5566c17dfa2de884fac7a6dc4f978f593d1efa8c4d0e"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1250` |
| `counts.false_positives` | `2642` |
| `counts.false_negatives` | `1387` |
| `metrics.recall` | `0.47402351156617367` |
| `metrics.precision` | `0.3211716341212744` |
| `metrics.frauds_per_1000_reviews` | `321.1716341212744` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.47402351156617367` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.478277933479642` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.45203696655900066` |
| `metrics.fraud_value_exposed` | `220699.084` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `43` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `30` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `76` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `49` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `41` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `54` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `75` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `35` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `40` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `35` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `54` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `26` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `50` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `52` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `59` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `41` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `39` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `32` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `50` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `30` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `35` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `33` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `34` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `38` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `46` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `75` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `45` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-b-seed_refit-s43-V2.json"` |
| `score.sha256` | `"b82d1a99f64a73cbea6c5566c17dfa2de884fac7a6dc4f978f593d1efa8c4d0e"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1542` |
| `counts.false_positives` | `6242` |
| `counts.false_negatives` | `1095` |
| `metrics.recall` | `0.5847554038680318` |
| `metrics.precision` | `0.19809866392600206` |
| `metrics.frauds_per_1000_reviews` | `198.09866392600205` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5847554038680318` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.846201829370243` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5858733044024202` |
| `metrics.fraud_value_exposed` | `166794.796` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `49` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `38` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `95` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `55` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `53` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `70` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `91` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `45` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `50` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `48` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `37` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `62` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `33` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `60` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `67` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `66` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `47` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `48` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `37` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `61` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `46` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `43` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `41` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `52` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `59` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `90` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `57` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"fe64c23cca877970e5f15699b3a4bfae2c893116359aa42b10953f247ef9acf6"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `534` |
| `counts.false_positives` | `250` |
| `counts.false_negatives` | `2103` |
| `metrics.recall` | `0.2025028441410694` |
| `metrics.precision` | `0.6811224489795918` |
| `metrics.frauds_per_1000_reviews` | `681.1224489795918` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.6811224489795918` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `20.100990225441713` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.1431002587021931` |
| `metrics.fraud_value_exposed` | `345127.274` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `11` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"fe64c23cca877970e5f15699b3a4bfae2c893116359aa42b10953f247ef9acf6"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `870` |
| `counts.false_positives` | `698` |
| `counts.false_negatives` | `1767` |
| `metrics.recall` | `0.32992036405005687` |
| `metrics.precision` | `0.5548469387755102` |
| `metrics.frauds_per_1000_reviews` | `554.8469387755102` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5548469387755102` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.374402149938472` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2757208460036138` |
| `metrics.fraud_value_exposed` | `291712.645` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `18` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `50` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `44` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `17` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `34` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `41` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `46` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"fe64c23cca877970e5f15699b3a4bfae2c893116359aa42b10953f247ef9acf6"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1316` |
| `counts.false_positives` | `2576` |
| `counts.false_negatives` | `1321` |
| `metrics.recall` | `0.49905195297686766` |
| `metrics.precision` | `0.3381294964028777` |
| `metrics.frauds_per_1000_reviews` | `338.1294964028777` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.49905195297686766` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.978731008367367` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.45848802780467834` |
| `metrics.fraud_value_exposed` | `218100.837` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `45` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `30` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `76` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `49` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `50` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `56` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `78` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `45` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `36` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `35` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `53` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `24` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `46` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `51` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `61` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `44` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `39` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `35` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `45` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `37` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `38` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `36` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `37` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `46` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `59` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `79` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `53` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-b-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-b"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-b-seed_refit-s44-V2.json"` |
| `score.sha256` | `"fe64c23cca877970e5f15699b3a4bfae2c893116359aa42b10953f247ef9acf6"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1572` |
| `counts.false_positives` | `6212` |
| `counts.false_negatives` | `1065` |
| `metrics.recall` | `0.59613196814562` |
| `metrics.precision` | `0.20195272353545735` |
| `metrics.frauds_per_1000_reviews` | `201.95272353545735` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.59613196814562` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.959941164571999` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5583048296197342` |
| `metrics.fraud_value_exposed` | `177898.35` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `53` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `35` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `90` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `53` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `56` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `68` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `98` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `54` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `45` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `39` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `41` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `60` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `32` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `61` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `75` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `73` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `50` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `52` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `42` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `56` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `38` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `46` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `42` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `46` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `54` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `66` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `86` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `61` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-same_seed_repeat-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"37b6d7e196e986c237e40975ca94312ad25c17d583e3e347e747dd263a0e63c3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `597` |
| `counts.false_positives` | `187` |
| `counts.false_negatives` | `2040` |
| `metrics.recall` | `0.22639362912400454` |
| `metrics.precision` | `0.7614795918367347` |
| `metrics.frauds_per_1000_reviews` | `761.4795918367347` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7614795918367347` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `22.472455364398318` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.20173802628269047` |
| `metrics.fraud_value_exposed` | `321510.167` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `24` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-same_seed_repeat-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"37b6d7e196e986c237e40975ca94312ad25c17d583e3e347e747dd263a0e63c3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `924` |
| `counts.false_positives` | `644` |
| `counts.false_negatives` | `1713` |
| `metrics.recall` | `0.3503981797497156` |
| `metrics.precision` | `0.5892857142857143` |
| `metrics.frauds_per_1000_reviews` | `589.2857142857143` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5892857142857143` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `17.39074435234845` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.3243901165987744` |
| `metrics.fraud_value_exposed` | `272110.477` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

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
| `148` | `3060` | `86` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `55` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-same_seed_repeat-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"37b6d7e196e986c237e40975ca94312ad25c17d583e3e347e747dd263a0e63c3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1316` |
| `counts.false_positives` | `2576` |
| `counts.false_negatives` | `1321` |
| `metrics.recall` | `0.49905195297686766` |
| `metrics.precision` | `0.3381294964028777` |
| `metrics.frauds_per_1000_reviews` | `338.1294964028777` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.49905195297686766` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.978731008367367` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.49680709032050346` |
| `metrics.fraud_value_exposed` | `202667.347` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `44` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `33` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `74` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `52` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `51` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `57` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `78` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `37` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `43` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `37` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `29` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `53` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `21` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `56` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `53` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `43` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `44` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `38` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `36` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `34` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `39` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `35` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `40` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `45` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `60` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `82` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `49` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-same_seed_repeat-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"same_seed_repeat"` |
| `seed` | `42` |
| `score.path` | `"records/score-RES-u-same_seed_repeat-V2.json"` |
| `score.sha256` | `"37b6d7e196e986c237e40975ca94312ad25c17d583e3e347e747dd263a0e63c3"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1580` |
| `counts.false_positives` | `6204` |
| `counts.false_negatives` | `1057` |
| `metrics.recall` | `0.5991657186196435` |
| `metrics.precision` | `0.20298047276464543` |
| `metrics.frauds_per_1000_reviews` | `202.9804727646454` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5991657186196435` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.990271653959134` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.6095603822562289` |
| `metrics.fraud_value_exposed` | `157254.524` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `51` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `37` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `92` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `59` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `61` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `69` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `97` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `54` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `46` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `48` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `62` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `68` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `68` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `63` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `50` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `49` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `48` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `50` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `42` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `45` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `43` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `54` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `66` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `91` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `58` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s43-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"7f2eeac675d2f686596ad9c9cfce2534a2e068cab6113235935eaad2bc14d888"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `564` |
| `counts.false_positives` | `220` |
| `counts.false_negatives` | `2073` |
| `metrics.recall` | `0.21387940841865757` |
| `metrics.precision` | `0.7193877551020408` |
| `metrics.frauds_per_1000_reviews` | `719.3877551020408` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7193877551020408` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `21.230259339230575` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.21227466174352322` |
| `metrics.fraud_value_exposed` | `317266.403` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `10` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `25` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s43-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"7f2eeac675d2f686596ad9c9cfce2534a2e068cab6113235935eaad2bc14d888"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `850` |
| `counts.false_positives` | `718` |
| `counts.false_negatives` | `1787` |
| `metrics.recall` | `0.3223359878649981` |
| `metrics.precision` | `0.5420918367346939` |
| `metrics.frauds_per_1000_reviews` | `542.0918367346939` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5420918367346939` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `15.997979112008853` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.31021254340309806` |
| `metrics.fraud_value_exposed` | `277820.675` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `45` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `35` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `17` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `20` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `37` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `39` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `51` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s43-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"7f2eeac675d2f686596ad9c9cfce2534a2e068cab6113235935eaad2bc14d888"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1246` |
| `counts.false_positives` | `2646` |
| `counts.false_negatives` | `1391` |
| `metrics.recall` | `0.4725066363291619` |
| `metrics.precision` | `0.32014388489208634` |
| `metrics.frauds_per_1000_reviews` | `320.1438848920863` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.4725066363291619` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.447947444092508` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.45607115319837793` |
| `metrics.fraud_value_exposed` | `219074.264` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `43` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `27` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `72` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `48` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `44` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `51` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `76` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `38` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `37` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `35` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `32` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `53` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `22` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `48` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `49` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `55` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `43` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `39` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `41` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `46` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `33` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `34` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `33` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `28` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `48` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `54` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `79` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `38` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s43-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `43` |
| `score.path` | `"records/score-RES-u-seed_refit-s43-V2.json"` |
| `score.sha256` | `"7f2eeac675d2f686596ad9c9cfce2534a2e068cab6113235935eaad2bc14d888"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1517` |
| `counts.false_positives` | `6267` |
| `counts.false_negatives` | `1120` |
| `metrics.recall` | `0.5752749336367083` |
| `metrics.precision` | `0.1948869475847893` |
| `metrics.frauds_per_1000_reviews` | `194.8869475847893` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5752749336367083` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.751419050035447` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5634463406797299` |
| `metrics.fraud_value_exposed` | `175827.541` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `55` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `32` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `89` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `55` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `58` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `65` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `89` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `45` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `46` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `44` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `38` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `60` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `28` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `70` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `68` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `64` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `49` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `50` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `56` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `37` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `39` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `37` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `40` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `60` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `61` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `88` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `48` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s44-V2-R3-c01.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"6b8472422a3678635242ad3454c3c8a0f3e5c183a9793d61930140d849d709df"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.01` |
| `k` | `28` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `784` |
| `counts.true_positives` | `560` |
| `counts.false_positives` | `224` |
| `counts.false_negatives` | `2077` |
| `metrics.recall` | `0.2123625331816458` |
| `metrics.precision` | `0.7142857142857143` |
| `metrics.frauds_per_1000_reviews` | `714.2857142857143` |
| `metrics.oracle_ceiling` | `0.29730754645430413` |
| `metrics.normalised_recall` | `0.7142857142857143` |
| `metrics.random_review_recall` | `0.01007427205674488` |
| `metrics.lift` | `21.079690124058725` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.29511566173682213` |
| `metrics.amount_weighted_recall` | `0.16364547926734155` |
| `metrics.fraud_value_exposed` | `336852.425` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `28` | `26` | `28` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `28` | `27` | `28` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `28` | `15` | `28` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `28` | `12` | `28` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `28` | `19` | `28` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `28` | `20` | `28` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `28` | `23` | `28` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `28` | `17` | `28` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `28` | `14` | `28` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `28` | `22` | `28` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `28` | `16` | `28` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `28` | `18` | `28` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `28` | `28` | `28` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `28` | `21` | `28` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s44-V2-R3-c02.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"6b8472422a3678635242ad3454c3c8a0f3e5c183a9793d61930140d849d709df"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.02` |
| `k` | `56` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `1568` |
| `counts.true_positives` | `878` |
| `counts.false_positives` | `690` |
| `counts.false_negatives` | `1759` |
| `metrics.recall` | `0.3329541145240804` |
| `metrics.precision` | `0.5599489795918368` |
| `metrics.frauds_per_1000_reviews` | `559.9489795918367` |
| `metrics.oracle_ceiling` | `0.5946150929086083` |
| `metrics.normalised_recall` | `0.5599489795918368` |
| `metrics.random_review_recall` | `0.02014854411348976` |
| `metrics.lift` | `16.52497136511032` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `0.5902313234736443` |
| `metrics.amount_weighted_recall` | `0.2829224757155034` |
| `metrics.fraud_value_exposed` | `288812.097` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `56` | `21` | `56` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `56` | `48` | `56` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `56` | `26` | `56` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `56` | `47` | `56` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `56` | `28` | `56` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `56` | `23` | `56` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `56` | `15` | `56` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `56` | `32` | `56` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `56` | `38` | `56` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `56` | `30` | `56` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `56` | `31` | `56` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `56` | `29` | `56` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `56` | `24` | `56` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `56` | `27` | `56` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `56` | `25` | `56` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `56` | `36` | `56` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `56` | `53` | `56` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `56` | `33` | `56` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s44-V2-R3-c05.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"6b8472422a3678635242ad3454c3c8a0f3e5c183a9793d61930140d849d709df"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.05` |
| `k` | `139` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `3892` |
| `counts.true_positives` | `1297` |
| `counts.false_positives` | `2595` |
| `counts.false_negatives` | `1340` |
| `metrics.recall` | `0.4918467956010618` |
| `metrics.precision` | `0.33324768756423434` |
| `metrics.frauds_per_1000_reviews` | `333.24768756423435` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.4918467956010618` |
| `metrics.random_review_recall` | `0.05001156485312636` |
| `metrics.lift` | `9.834661183778477` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `1.4755783086841108` |
| `metrics.amount_weighted_recall` | `0.4393605799527764` |
| `metrics.fraud_value_exposed` | `225804.66` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `139` | `44` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `139` | `32` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `139` | `72` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `139` | `53` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `139` | `48` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `139` | `49` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `139` | `75` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `139` | `34` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `139` | `41` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `139` | `36` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `139` | `30` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `139` | `49` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `139` | `24` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `139` | `50` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `139` | `54` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `139` | `56` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `139` | `43` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `139` | `45` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `139` | `41` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `139` | `39` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `139` | `38` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `139` | `35` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `139` | `33` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `139` | `37` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `139` | `50` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `139` | `56` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `139` | `83` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `139` | `50` | `94` | `0` | `0` | `1.0` | — |

### `evaluation-RES-u-seed_refit-s44-V2-R3-c10.json`

| Field | Value |
|---|---|
| `record` | `"evaluation"` |
| `schema_version` | `1` |
| `configuration` | `"RES-u"` |
| `model_role` | `"seed_refit"` |
| `seed` | `44` |
| `score.path` | `"records/score-RES-u-seed_refit-s44-V2.json"` |
| `score.sha256` | `"6b8472422a3678635242ad3454c3c8a0f3e5c183a9793d61930140d849d709df"` |
| `partition` | `"V2"` |
| `period` | `"primary"` |
| `days.first` | `127` |
| `days.last` | `154` |
| `policy` | `"R3"` |
| `capacity` | `0.1` |
| `k` | `278` |
| `threshold` | — |
| `cost_ratio` | — |
| `calibrator` | — |
| `calibration` | — |
| `counts.days` | `28` |
| `counts.transactions` | `77822` |
| `counts.frauds` | `2637` |
| `counts.selected` | `7784` |
| `counts.true_positives` | `1553` |
| `counts.false_positives` | `6231` |
| `counts.false_negatives` | `1084` |
| `metrics.recall` | `0.5889268107698142` |
| `metrics.precision` | `0.19951181911613566` |
| `metrics.frauds_per_1000_reviews` | `199.51181911613565` |
| `metrics.oracle_ceiling` | `1.0` |
| `metrics.normalised_recall` | `0.5889268107698142` |
| `metrics.random_review_recall` | `0.10002312970625273` |
| `metrics.lift` | `5.887906252277554` |
| `metrics.utilisation` | `1.0` |
| `metrics.capacity_to_prevalence` | `2.9511566173682215` |
| `metrics.amount_weighted_recall` | `0.5433597499454791` |
| `metrics.fraud_value_exposed` | `183917.671` |
| `cutoff_ties.days` | `0` |
| `cutoff_ties.group_sizes` | `[]` |

**`daily`** (28 entries)

| `day` | `transactions` | `frauds` | `selected` | `true_positives` | `oracle_reviewable` | `overflow` | `unused` | `load_ratio` | `cutoff_tied_group` |
|---|---|---|---|---|---|---|---|---|---|
| `127` | `3335` | `84` | `278` | `50` | `84` | `0` | `0` | `1.0` | — |
| `128` | `2507` | `69` | `278` | `42` | `69` | `0` | `0` | `1.0` | — |
| `129` | `2348` | `130` | `278` | `84` | `130` | `0` | `0` | `1.0` | — |
| `130` | `2807` | `95` | `278` | `59` | `95` | `0` | `0` | `1.0` | — |
| `131` | `2822` | `105` | `278` | `58` | `105` | `0` | `0` | `1.0` | — |
| `132` | `2752` | `108` | `278` | `60` | `108` | `0` | `0` | `1.0` | — |
| `133` | `2687` | `136` | `278` | `94` | `136` | `0` | `0` | `1.0` | — |
| `134` | `2937` | `103` | `278` | `45` | `103` | `0` | `0` | `1.0` | — |
| `135` | `2429` | `90` | `278` | `43` | `90` | `0` | `0` | `1.0` | — |
| `136` | `2048` | `73` | `278` | `39` | `73` | `0` | `0` | `1.0` | — |
| `137` | `2914` | `69` | `278` | `37` | `69` | `0` | `0` | `1.0` | — |
| `138` | `2846` | `112` | `278` | `60` | `112` | `0` | `0` | `1.0` | — |
| `139` | `2769` | `61` | `278` | `27` | `61` | `0` | `0` | `1.0` | — |
| `140` | `2629` | `103` | `278` | `71` | `103` | `0` | `0` | `1.0` | — |
| `141` | `3000` | `131` | `278` | `61` | `131` | `0` | `0` | `1.0` | — |
| `142` | `2391` | `103` | `278` | `62` | `103` | `0` | `0` | `1.0` | — |
| `143` | `2064` | `71` | `278` | `51` | `71` | `0` | `0` | `1.0` | — |
| `144` | `2718` | `90` | `278` | `53` | `90` | `0` | `0` | `1.0` | — |
| `145` | `2668` | `67` | `278` | `46` | `67` | `0` | `0` | `1.0` | — |
| `146` | `2990` | `98` | `278` | `54` | `98` | `0` | `0` | `1.0` | — |
| `147` | `2718` | `79` | `278` | `45` | `79` | `0` | `0` | `1.0` | — |
| `148` | `3060` | `86` | `278` | `51` | `86` | `0` | `0` | `1.0` | — |
| `149` | `2289` | `66` | `278` | `42` | `66` | `0` | `0` | `1.0` | — |
| `150` | `2209` | `69` | `278` | `45` | `69` | `0` | `0` | `1.0` | — |
| `151` | `3101` | `99` | `278` | `60` | `99` | `0` | `0` | `1.0` | — |
| `152` | `3896` | `120` | `278` | `67` | `120` | `0` | `0` | `1.0` | — |
| `153` | `3105` | `126` | `278` | `91` | `126` | `0` | `0` | `1.0` | — |
| `154` | `3783` | `94` | `278` | `56` | `94` | `0` | `0` | `1.0` | — |

### `device_comparison-Cat-b.json`

| Field | Value |
|---|---|
| `record` | `"device_comparison"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-b"` |
| `seed` | `42` |
| `gpu_score.path` | `"source/records/score-Cat-b-selected-V2.json"` |
| `gpu_score.sha256` | `"6c3b5a752fce19ff08c04deaf61c82e15335513096cd2d9c3fab40363e7c49df"` |
| `cpu_score.path` | `"records/score-Cat-b-cpu_refit-V2.json"` |
| `cpu_score.sha256` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `refit.path` | `"records/refit-Cat-b-cpu_refit.json"` |
| `refit.sha256` | `"0b6c5cbe1d9337e0dbbedb1cd4708497ef24f3170a2ec4ca7b3ab815baa300ef"` |
| `bootstrap.capacity` | `0.02` |
| `bootstrap.k` | `56` |
| `bootstrap.entity_key` | `"fine"` |
| `bootstrap.estimate` | `-0.012514220705346985` |
| `bootstrap.lower` | `-0.036189320063896475` |
| `bootstrap.upper` | `0.010899430270002473` |
| `bootstrap.level` | `0.95` |
| `bootstrap.replicates` | `10000` |
| `bootstrap.undefined_replicates` | `0` |
| `bootstrap.seed` | `42` |
| `bootstrap.days` | `28` |
| `bootstrap.entities` | `1353` |
| `bootstrap.frauds` | `2637` |

**`capacities`** (4 entries)

| `capacity` | `k` | `differences.recall` | `differences.precision` | `differences.frauds_per_1000_reviews` | `differences.oracle_ceiling` | `differences.normalised_recall` | `differences.random_review_recall` | `differences.lift` | `differences.utilisation` | `differences.capacity_to_prevalence` | `differences.amount_weighted_recall` | `differences.fraud_value_exposed` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0.01` | `28` | `-0.006067500948047033` | `-0.020408163265306145` | `-20.408163265306143` | `0.0` | `-0.020408163265306145` | `0.0` | `-0.6022768606873932` | `0.0` | `0.0` | `-0.022151915925566118` | `8921.966000000015` |
| `0.02` | `56` | `-0.012514220705346968` | `-0.021045918367346927` | `-21.045918367346985` | `0.0` | `-0.021045918367346927` | `0.0` | `-0.6210980125838752` | `0.0` | `0.0` | `-0.014508872474504386` | `5843.633000000031` |
| `0.05` | `139` | `-0.015547971179370568` | `-0.010534429599177775` | `-10.534429599177827` | `0.0` | `-0.015547971179370568` | `0.0` | `-0.3108875162181324` | `0.0` | `0.0` | `-0.05940651548478448` | `23926.73000000001` |
| `0.1` | `278` | `-0.0007584376185059138` | `-0.00025693730729700603` | `-0.25693730729702224` | `0.0` | `-0.0007584376185059138` | `0.0` | `-0.007582622346783197` | `0.0` | `0.0` | `-0.04407546910920179` | `17751.956000000006` |

### `device_comparison-Cat-u.json`

| Field | Value |
|---|---|
| `record` | `"device_comparison"` |
| `schema_version` | `1` |
| `configuration` | `"Cat-u"` |
| `seed` | `42` |
| `gpu_score.path` | `"source/records/score-Cat-u-selected-V2.json"` |
| `gpu_score.sha256` | `"a6f0ac75ca0ca7cfda36c28cdebf96c563a5097eb7b77a115e72fd9f8b224ec6"` |
| `cpu_score.path` | `"records/score-Cat-u-cpu_refit-V2.json"` |
| `cpu_score.sha256` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `refit.path` | `"records/refit-Cat-u-cpu_refit.json"` |
| `refit.sha256` | `"8e2a042dbd2e5c201048fb995dfcf562845b00fe1678944f0a645a3312f30ca3"` |
| `bootstrap.capacity` | `0.02` |
| `bootstrap.k` | `56` |
| `bootstrap.entity_key` | `"fine"` |
| `bootstrap.estimate` | `-0.004171406901782328` |
| `bootstrap.lower` | `-0.020524653721691113` |
| `bootstrap.upper` | `0.011077514009556387` |
| `bootstrap.level` | `0.95` |
| `bootstrap.replicates` | `10000` |
| `bootstrap.undefined_replicates` | `0` |
| `bootstrap.seed` | `42` |
| `bootstrap.days` | `28` |
| `bootstrap.entities` | `1353` |
| `bootstrap.frauds` | `2637` |

**`capacities`** (4 entries)

| `capacity` | `k` | `differences.recall` | `differences.precision` | `differences.frauds_per_1000_reviews` | `differences.oracle_ceiling` | `differences.normalised_recall` | `differences.random_review_recall` | `differences.lift` | `differences.utilisation` | `differences.capacity_to_prevalence` | `differences.amount_weighted_recall` | `differences.fraud_value_exposed` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0.01` | `28` | `-0.0011376564277588153` | `-0.003826530612244916` | `-3.8265306122449374` | `0.0` | `-0.003826530612244916` | `0.0` | `-0.11292691137888511` | `0.0` | `0.0` | `-0.00032262916167982714` | `129.9430000000284` |
| `0.02` | `56` | `-0.004171406901782304` | `-0.007015306122448939` | `-7.015306122448919` | `0.0` | `-0.007015306122448939` | `0.0` | `-0.20703267086129173` | `0.0` | `0.0` | `-0.006544796335223868` | `2636.0` |
| `0.05` | `139` | `0.008722032612817565` | `0.005909558067831444` | `5.909558067831426` | `0.0` | `0.008722032612817565` | `0.0` | `0.17440031397602418` | `0.0` | `0.0` | `-0.016545865848300367` | `6664.05799999999` |
| `0.1` | `278` | `-0.0068259385665528916` | `-0.0023124357656731653` | `-2.3124357656732` | `0.0` | `-0.0068259385665528916` | `0.0` | `-0.06824360112105321` | `0.0` | `0.0` | `-0.014690018831037599` | `5916.592000000004` |

### `run_end.json`

| Field | Value |
|---|---|
| `record` | `"run_end"` |
| `schema_version` | `1` |
| `finished` | `"2026-10-30T01:14:30.429503Z"` |
| `outcome` | `"completed"` |
| `error` | — |

**`records`** (185 entries)

| `path` | `sha256` |
|---|---|
| `"records/device_comparison-Cat-b.json"` | `"4909190c1d15f9d49f5ce568deacd5ce74e46c9b8006a8f8afe16f62db9b564b"` |
| `"records/device_comparison-Cat-u.json"` | `"9ed153b6c4c0cdfe8d453317beae4bd61bc93946676243a676339d183c2c40a9"` |
| `"records/evaluation-Cat-b-cpu_refit-V2-R3-c01.json"` | `"9631150a22a6d5986748c9dbd74bce9de68ec2f4c29560c1b7bcc723e32fdb73"` |
| `"records/evaluation-Cat-b-cpu_refit-V2-R3-c02.json"` | `"08a4d1732df17b3816d30c9df480bb4067ed57dd0f2b8cb46da75f71aa1f0dd5"` |
| `"records/evaluation-Cat-b-cpu_refit-V2-R3-c05.json"` | `"fbb65cee35bf8dbf464c9d0c0b3dbee5dca289bcfc0d3cbd1df72f7944b5cc67"` |
| `"records/evaluation-Cat-b-cpu_refit-V2-R3-c10.json"` | `"3e978a508da0d59a9853515c046c0d213e4e0f580473e28e4813c24458c6260c"` |
| `"records/evaluation-Cat-b-same_seed_repeat-V2-R3-c01.json"` | `"ee6dbfa4cc93ce5ed43ad858b5782c03d51901f0a495121d9908874375d6531e"` |
| `"records/evaluation-Cat-b-same_seed_repeat-V2-R3-c02.json"` | `"30efca2b9652597d8487d32ae455260b54f34a03ddd014bd50297f74169609da"` |
| `"records/evaluation-Cat-b-same_seed_repeat-V2-R3-c05.json"` | `"4106840ffc39b98c93bc35e93e6cd5c90fc6493c77a9f93d34ac061cd736d450"` |
| `"records/evaluation-Cat-b-same_seed_repeat-V2-R3-c10.json"` | `"3cc094f4ed1c941fc71fe5fe72dcdbefb09c21ab2fb2dabd2df4667c9decf49f"` |
| `"records/evaluation-Cat-b-seed_refit-s43-V2-R3-c01.json"` | `"9e563bf6cf94905a13744d9c7ae8ef52f735c927960f2f60c9a8f594d83aa661"` |
| `"records/evaluation-Cat-b-seed_refit-s43-V2-R3-c02.json"` | `"a024fece50ce14e1a6da0a547d17936b9dbfc0af7457165076b139bf0a3be0c6"` |
| `"records/evaluation-Cat-b-seed_refit-s43-V2-R3-c05.json"` | `"3d9de7ecc7892f50a706aaa8e16077e498abafcd916eeca9a49da3c3723b6af3"` |
| `"records/evaluation-Cat-b-seed_refit-s43-V2-R3-c10.json"` | `"f41be94be4bf2ddd3b86d65d32317169784e19baea68c78e6776413b086b6fee"` |
| `"records/evaluation-Cat-b-seed_refit-s44-V2-R3-c01.json"` | `"6b015b3d7b78d8033bdb9a0a149f0acfd980a59a4718e27cf5425ca36a6e82d9"` |
| `"records/evaluation-Cat-b-seed_refit-s44-V2-R3-c02.json"` | `"fd727a3fb1141df6468ee6c312e4fac6b945bcc6076d6f381c841d7f251180ee"` |
| `"records/evaluation-Cat-b-seed_refit-s44-V2-R3-c05.json"` | `"3073f897b20d1613f287418540b150cafed4b7e113bc5c20ef3e0b020ddca9a0"` |
| `"records/evaluation-Cat-b-seed_refit-s44-V2-R3-c10.json"` | `"7fed853de23a43a18552cc2efc3d6005cf2cb87c7230d3177ac47ee4f2f3a63c"` |
| `"records/evaluation-Cat-u-cpu_refit-V2-R3-c01.json"` | `"f24e8c2cbc43facbdba1744181e0924cdd110232eccac75e5d0a8ffe70ae3b98"` |
| `"records/evaluation-Cat-u-cpu_refit-V2-R3-c02.json"` | `"58e91283a292a570ab0d865396835a20d10dcf40e79e6b41536905a6aa644103"` |
| `"records/evaluation-Cat-u-cpu_refit-V2-R3-c05.json"` | `"c878b6cb1cc8f7408814f55d3398b98b43fb03263eeac8e9ac21836043bfbfaf"` |
| `"records/evaluation-Cat-u-cpu_refit-V2-R3-c10.json"` | `"23aec3a4eea434a8ad8f024b179a05f7383ec713dc664aec020b4fc78ec6a93c"` |
| `"records/evaluation-Cat-u-same_seed_repeat-V2-R3-c01.json"` | `"71267b97725d5d3fa893a00ae9a85b4ac1e5c3e909e21324dc3e54f7672bc166"` |
| `"records/evaluation-Cat-u-same_seed_repeat-V2-R3-c02.json"` | `"7ce3264df5f5008d7a64ea9e86f0af8330705501266b37fe2fb2e526f39d06c2"` |
| `"records/evaluation-Cat-u-same_seed_repeat-V2-R3-c05.json"` | `"41d39f5ca8bd9a42e85d2c8827f63f6529355540d6ebaf131730e8dacaeaebe0"` |
| `"records/evaluation-Cat-u-same_seed_repeat-V2-R3-c10.json"` | `"f7edc946268e58b1a8eebf72363fcde85752e72822279419bdf10ae518d21d52"` |
| `"records/evaluation-Cat-u-seed_refit-s43-V2-R3-c01.json"` | `"e58733a0be8451ad3992ebe1be914b929af9d46eb6dea3dc11c69f7777a10efe"` |
| `"records/evaluation-Cat-u-seed_refit-s43-V2-R3-c02.json"` | `"974669cee1f50ff37b356a16225b05332d861387f6a9fb98112575fe0f9dd5d5"` |
| `"records/evaluation-Cat-u-seed_refit-s43-V2-R3-c05.json"` | `"e818ace6dbf03d2a4abd974c38915ff9d13d89874788cba38a9373dfd33d2900"` |
| `"records/evaluation-Cat-u-seed_refit-s43-V2-R3-c10.json"` | `"1d99600e3b135c37ef5f2026f32fa8732063fa47042798143f38020351ecf856"` |
| `"records/evaluation-Cat-u-seed_refit-s44-V2-R3-c01.json"` | `"d7c4cd8f92e0bdb2cd7214e6922bedf0bca9b1c4c07374f889d2f338282ffeb2"` |
| `"records/evaluation-Cat-u-seed_refit-s44-V2-R3-c02.json"` | `"4a664a0e1e1872ebfba8b8406d2c1839cf9398c9ea723c504ff810761a81c09e"` |
| `"records/evaluation-Cat-u-seed_refit-s44-V2-R3-c05.json"` | `"e9d9ccddeb7b0df3fe088447de12d263ab66028e92ca06761642dc8b99a72642"` |
| `"records/evaluation-Cat-u-seed_refit-s44-V2-R3-c10.json"` | `"f30c6ef590180e52aa199eb186700d1e973c146374225b6bcd261e7bc6c1af5a"` |
| `"records/evaluation-FTT-b-same_seed_repeat-V2-R3-c01.json"` | `"d9d584f4b2170e56b415dd80abd607f8343a0139b25c7e67a2b32187322b9168"` |
| `"records/evaluation-FTT-b-same_seed_repeat-V2-R3-c02.json"` | `"8714aa04aa3a5fa4295f3f5573f96ea85d79d10611d9f5380ec0769cf1607735"` |
| `"records/evaluation-FTT-b-same_seed_repeat-V2-R3-c05.json"` | `"233b13127789b2c76c335f40dfbe8de431a29dd14b1bf193e6a4f5d4a4ae1370"` |
| `"records/evaluation-FTT-b-same_seed_repeat-V2-R3-c10.json"` | `"32eedab923ea36dbc909d977bd10f8e614b11e640f2ca5856949ebec2d05f55e"` |
| `"records/evaluation-FTT-b-seed_refit-s43-V2-R3-c01.json"` | `"d007a6779a2cf6d3aa7cd1cfbaf532224095f782402dec2e26621075551c6ef0"` |
| `"records/evaluation-FTT-b-seed_refit-s43-V2-R3-c02.json"` | `"01f4c2851b8dc9a9ae789d0c3fb0960629e27e6359aba51be305cd5d1e9b63e0"` |
| `"records/evaluation-FTT-b-seed_refit-s43-V2-R3-c05.json"` | `"228bac803338718de5091db85e6e07c581517ced688f52f74d693c36d910ac5e"` |
| `"records/evaluation-FTT-b-seed_refit-s43-V2-R3-c10.json"` | `"9025a458d5f8316e989013ed02915034a57d98d9090e3c7fd47964a5db9950a9"` |
| `"records/evaluation-FTT-b-seed_refit-s44-V2-R3-c01.json"` | `"befd170dc3bdfc987c0373ac562ed733a72a207bf69830b95eaf53bd709bb47e"` |
| `"records/evaluation-FTT-b-seed_refit-s44-V2-R3-c02.json"` | `"53212945cd5f027a3ed7f7e01e3f7d9a053d7e9aba5b118a3236bb4e48ecaa54"` |
| `"records/evaluation-FTT-b-seed_refit-s44-V2-R3-c05.json"` | `"584e96b2387b8fa449b2f2ed97ff8873f8a9932b7781da52e5a2ddb068684c4e"` |
| `"records/evaluation-FTT-b-seed_refit-s44-V2-R3-c10.json"` | `"b8aa884105c9eb9e275d7da9edcd8d40ee8b08e67d047e8ae2326f93924955ac"` |
| `"records/evaluation-FTT-u-same_seed_repeat-V2-R3-c01.json"` | `"1ec5f9966f76fabf6d95e517f6cd0865f8df8386375553505005ddbf3d9e685e"` |
| `"records/evaluation-FTT-u-same_seed_repeat-V2-R3-c02.json"` | `"1fed413567aeff9c0b44c2dd04bb0b327ac37f3bef879ab8757f5b6ad641c2d6"` |
| `"records/evaluation-FTT-u-same_seed_repeat-V2-R3-c05.json"` | `"1a714d2e11c38a68009c63d76b4bd1e9c608b5ad698d427b5936bde1792b1a5c"` |
| `"records/evaluation-FTT-u-same_seed_repeat-V2-R3-c10.json"` | `"01b5c2a5ce7188146efb233c67d3014184ae0e1c40d38b4ebbed55f90bb0368a"` |
| `"records/evaluation-FTT-u-seed_refit-s43-V2-R3-c01.json"` | `"9dc8566d3728ca5362d1a6539803fd677304e95c6a48091b9f1f489789ee215e"` |
| `"records/evaluation-FTT-u-seed_refit-s43-V2-R3-c02.json"` | `"b2216423dca70005ba439c9d3008c1bf6c46b0e305685a46b9368cc5dec27c72"` |
| `"records/evaluation-FTT-u-seed_refit-s43-V2-R3-c05.json"` | `"fed314fe98a95a245593e3db418716b796205bf1acbdcbbbbddac454eea4d29a"` |
| `"records/evaluation-FTT-u-seed_refit-s43-V2-R3-c10.json"` | `"98648db1eaa3a068a96dd61bc0c44a35c1c96632860db986c2eafd938078fab5"` |
| `"records/evaluation-FTT-u-seed_refit-s44-V2-R3-c01.json"` | `"e0662688ad12205545d330c292d3dbf84674cffaa3bff8405596df97294388ca"` |
| `"records/evaluation-FTT-u-seed_refit-s44-V2-R3-c02.json"` | `"46aa1af7e79ba170d81be942737f9fbc8cbc5302f43544eff81567b3145d052c"` |
| `"records/evaluation-FTT-u-seed_refit-s44-V2-R3-c05.json"` | `"51a4a680bfcbd5ec9908b94d02da43d0ca04250d6557f5c0bdd45c34279e9551"` |
| `"records/evaluation-FTT-u-seed_refit-s44-V2-R3-c10.json"` | `"f5685057f4e36cc7e8078fdc7f63aaf839a149de1bd46c26a26d2ade1caab41f"` |
| `"records/evaluation-LGB-b-same_seed_repeat-V2-R3-c01.json"` | `"d55bade25673cd341079629b455f85f76b9544cbfe6f110963debac1d99f0c21"` |
| `"records/evaluation-LGB-b-same_seed_repeat-V2-R3-c02.json"` | `"11c07c9f9a7df8bc96a960110c3634b5f0faa7a133dadf01c5b4bc917f0980a5"` |
| `"records/evaluation-LGB-b-same_seed_repeat-V2-R3-c05.json"` | `"72395542e8c3515ac8bc46da1e1224994bd35e12dba1bbd56aede3c6c8398595"` |
| `"records/evaluation-LGB-b-same_seed_repeat-V2-R3-c10.json"` | `"5f25fe81e828ab762a42fc1a56c2d5967b80a83a7d0a4e1d32531e32d15765d2"` |
| `"records/evaluation-LGB-b-seed_refit-s43-V2-R3-c01.json"` | `"ae4624c2c6c9efe14c12b2a39616d42ae56db9b3e96a78321bd8cc8eb4486055"` |
| `"records/evaluation-LGB-b-seed_refit-s43-V2-R3-c02.json"` | `"1c04fe71e2cac3dd48ec9228c2166ec77864bfdb4ff4517b761a88c4d5b3140a"` |
| `"records/evaluation-LGB-b-seed_refit-s43-V2-R3-c05.json"` | `"ffde5f9a67d89bd8c34096374174f9fb7ef6c53298c469cd8a9c5f29c6d9208e"` |
| `"records/evaluation-LGB-b-seed_refit-s43-V2-R3-c10.json"` | `"13f12212e5c16ff8471349ebc8eea724cbaa06c8b431787168a3eedb621a4a8f"` |
| `"records/evaluation-LGB-b-seed_refit-s44-V2-R3-c01.json"` | `"50c8914c0a11afe55659a2d30583138350aa5e0a4b4b6d42f580adf61c0f98fd"` |
| `"records/evaluation-LGB-b-seed_refit-s44-V2-R3-c02.json"` | `"71f9e0f4d550de660ce1378c33e95c0e75c6cbbb64fa277cccca7544214143a0"` |
| `"records/evaluation-LGB-b-seed_refit-s44-V2-R3-c05.json"` | `"750fe668ec8333594376cd8918da550339eaee1964d500bc9edbb55f8b3c3542"` |
| `"records/evaluation-LGB-b-seed_refit-s44-V2-R3-c10.json"` | `"292b3be1579b9a44d6c58210fef8ee7639ac2d945780c8116d38ff6620562d9c"` |
| `"records/evaluation-LGB-u-same_seed_repeat-V2-R3-c01.json"` | `"7a7f12e0806fa18e0b04dc27529bbd9c0623389325527d99bc1d95487a265960"` |
| `"records/evaluation-LGB-u-same_seed_repeat-V2-R3-c02.json"` | `"24bca53b00b9b6ecddf79fe4a661b24e52e3a04fcfac69cd06a7810ef8da7b06"` |
| `"records/evaluation-LGB-u-same_seed_repeat-V2-R3-c05.json"` | `"7cc612451f6bcd7b0e1f47ca04aeee4e26f5527ef5d9ccf7010d8615821e2242"` |
| `"records/evaluation-LGB-u-same_seed_repeat-V2-R3-c10.json"` | `"7c532d908dd62648e839819c90f2c2b6d9220f6b96d4d4f610c4f4d6e1c886f5"` |
| `"records/evaluation-LGB-u-seed_refit-s43-V2-R3-c01.json"` | `"796d7724b0fcff199f14d374f412fdb7165a9acd033df3ae7db872342bfd8d09"` |
| `"records/evaluation-LGB-u-seed_refit-s43-V2-R3-c02.json"` | `"f841044bca026d0a1632d5bbc7a63988a3172db53ca181397a0141234528ddce"` |
| `"records/evaluation-LGB-u-seed_refit-s43-V2-R3-c05.json"` | `"9e4be305e0212f47b3309b8611abee3e62a849dcd363eb32e35454493fff1a92"` |
| `"records/evaluation-LGB-u-seed_refit-s43-V2-R3-c10.json"` | `"5b0b6264ae0d490f9baf89808b21645b33ab8de805f414e96a22b857fdcb5415"` |
| `"records/evaluation-LGB-u-seed_refit-s44-V2-R3-c01.json"` | `"50c148e4000959a82c890a4666f0abfdc472f106ac0a96ad19284f98d0078a14"` |
| `"records/evaluation-LGB-u-seed_refit-s44-V2-R3-c02.json"` | `"058db157644eddb90dea8299d8bf64eb9cc0ede601e15d39504d005c9e2e09ab"` |
| `"records/evaluation-LGB-u-seed_refit-s44-V2-R3-c05.json"` | `"98b28b49d0aaf7d216ba4be7154274f01c32bcdd0250368f2c365ce218b2b281"` |
| `"records/evaluation-LGB-u-seed_refit-s44-V2-R3-c10.json"` | `"ecc2ded07506b2db86c9fb7c5c35ff6da54107439aefdde32bb00da7b26084b1"` |
| `"records/evaluation-RES-b-same_seed_repeat-V2-R3-c01.json"` | `"716c75860e5bd4f40c5467d2fbffec70ac2e9ba87a18a4cbd03dc3a793a8cffd"` |
| `"records/evaluation-RES-b-same_seed_repeat-V2-R3-c02.json"` | `"4d761cd12006107499298f6e9a0ed07d9e2577ac21b2792efc2c00a77716dec9"` |
| `"records/evaluation-RES-b-same_seed_repeat-V2-R3-c05.json"` | `"d83e982824a3a818d501b7e14f14d3a120115a60f2faf178ab05a6366948d1c1"` |
| `"records/evaluation-RES-b-same_seed_repeat-V2-R3-c10.json"` | `"2c775c0f72451bf5547f1afc36c57b8a2a3bb2414dc49fb3a19e05b167a1b6b2"` |
| `"records/evaluation-RES-b-seed_refit-s43-V2-R3-c01.json"` | `"4fa9ce93ea80e231530119166d8c300732eff71be20eb2e23fd3f2b5788f0892"` |
| `"records/evaluation-RES-b-seed_refit-s43-V2-R3-c02.json"` | `"b9fcc1aefa3bf8172c395cbf07c3d33f22e9c3628b84851f172962170d625537"` |
| `"records/evaluation-RES-b-seed_refit-s43-V2-R3-c05.json"` | `"952bbfbabf1a69724d11612346548fdc347086d77f6fd2e29fe1492e8670c136"` |
| `"records/evaluation-RES-b-seed_refit-s43-V2-R3-c10.json"` | `"78f27c011281d50fe56eba310a4f00731749e6378fb4c66cdd00298111797941"` |
| `"records/evaluation-RES-b-seed_refit-s44-V2-R3-c01.json"` | `"6451aa658c9a63137b459bef20686687b727a1b43350a9f4dfff1b41af0d758c"` |
| `"records/evaluation-RES-b-seed_refit-s44-V2-R3-c02.json"` | `"b37664ef4fd9f3303e6d8d27a9f8addd7ea3f871c186e28a047ea78793613897"` |
| `"records/evaluation-RES-b-seed_refit-s44-V2-R3-c05.json"` | `"4f409c0ea1acfca8f14417fb74b4cb181c812ecb032df07e08b63af7d2281e69"` |
| `"records/evaluation-RES-b-seed_refit-s44-V2-R3-c10.json"` | `"8d93f4df1cfda6f58138288f07170570a03b439c4561af9dca17880c508fc423"` |
| `"records/evaluation-RES-u-same_seed_repeat-V2-R3-c01.json"` | `"4e3f4f817f71f7d41d57ac455d23fa86f7fc8f421e54e442bc70d61bca51b529"` |
| `"records/evaluation-RES-u-same_seed_repeat-V2-R3-c02.json"` | `"3106d1c980bbc4f32853ec0c85009eef7c9bd4f94c7ef78706096cb52b1d814a"` |
| `"records/evaluation-RES-u-same_seed_repeat-V2-R3-c05.json"` | `"00e464ef3534e1afcf36135038ec7e796baf0767a484ec23a3c6146a3b39e4be"` |
| `"records/evaluation-RES-u-same_seed_repeat-V2-R3-c10.json"` | `"a6c60321d3f37e11e01ac3d2ccb62efcf96cff46ce52953c9a925ad41c23e503"` |
| `"records/evaluation-RES-u-seed_refit-s43-V2-R3-c01.json"` | `"510b5c71895e8738e938ccb21550bb981bb50e0ab43706eb799845832492ceb3"` |
| `"records/evaluation-RES-u-seed_refit-s43-V2-R3-c02.json"` | `"987846505749f3dbedd8ad14d47dda265a8e2e7335f41b8891ae779d187fc6f1"` |
| `"records/evaluation-RES-u-seed_refit-s43-V2-R3-c05.json"` | `"c993f22979b2540acac185c9f507ea45f9574cb9aaad8a5c003b3fe4d47693cc"` |
| `"records/evaluation-RES-u-seed_refit-s43-V2-R3-c10.json"` | `"65e47b27c02df08063d4975e53f2a7089c227b55a3fca7b369d0a2d130414c54"` |
| `"records/evaluation-RES-u-seed_refit-s44-V2-R3-c01.json"` | `"02e793afefc1c01277a4d42b2a8ea7a7b799cb8b0db1cd5a075b2bb3a13010a0"` |
| `"records/evaluation-RES-u-seed_refit-s44-V2-R3-c02.json"` | `"b86d9500d49741a1b60edbd7d0a5b23b07aa9cbf55993200a2dd7f17a2dd1c37"` |
| `"records/evaluation-RES-u-seed_refit-s44-V2-R3-c05.json"` | `"e9aa8d3b6fece32fec0d1212025cc3be5dd59486fde2ab09108705f055580ebe"` |
| `"records/evaluation-RES-u-seed_refit-s44-V2-R3-c10.json"` | `"7f86c65afa84c0a5a4cf6355bfd968e7823869481900b93a7f5ef4807cbaa61f"` |
| `"records/fit-Cat-b-cpu_refit.json"` | `"2fa268c84d7dd07a277bef40fb9024401af4bcef9e4ddffa285a8d19df0dff40"` |
| `"records/fit-Cat-b-same_seed_repeat.json"` | `"c42a09c43338bb0c7a252b7879aafa7f20c170fbaa65dbb54b48c4227bfa35f4"` |
| `"records/fit-Cat-b-seed_refit-s43.json"` | `"8230ecb5e5d89f0afaf30273682cc51861470e53c32f906387e957942a7fff20"` |
| `"records/fit-Cat-b-seed_refit-s44.json"` | `"1f926d5edbea5363372167644a63f4aae429a3215495d2f3260a4d40f2576369"` |
| `"records/fit-Cat-u-cpu_refit.json"` | `"966657c178f5ef32e72c38909ceb8e94b1e26d4fa1d48130d8942aea2751c075"` |
| `"records/fit-Cat-u-same_seed_repeat.json"` | `"b09be1ae6cb53e2c4048dd8fe3ab4559a64d83c7c936aab3a4c78e686bfbec8c"` |
| `"records/fit-Cat-u-seed_refit-s43.json"` | `"3df2d1f16f8d169e0d3368cb4cdba274bd22b70e0f906e9f9af95fb3181f630e"` |
| `"records/fit-Cat-u-seed_refit-s44.json"` | `"84f0b0e4d9cd3ddc5590da3bcb594bcc67331e4541a55e0f215f34843d9d18a1"` |
| `"records/fit-FTT-b-same_seed_repeat.json"` | `"52e1240d356c1728b9900adc06f7bd1740fb783d2e6bf15301c5b8efbc2d671a"` |
| `"records/fit-FTT-b-seed_refit-s43.json"` | `"f2ad1044a6d60386bdb4bbcf8117192aafde913d5eae3b935c0ef8ea9eef8e6d"` |
| `"records/fit-FTT-b-seed_refit-s44.json"` | `"efa40b3b0f13baa3fac5442695d72d195f438f6a1ae14e4b67f968dce3530fb8"` |
| `"records/fit-FTT-u-same_seed_repeat.json"` | `"bb4be736b9192bc198a08d360aabcd35ef8e6f5d8ac0545d694bb611d6ac209c"` |
| `"records/fit-FTT-u-seed_refit-s43.json"` | `"9332e0e0af09e7bd4c2dfde4643a4c4a157ebdf9dc04f57685733011f29b631e"` |
| `"records/fit-FTT-u-seed_refit-s44.json"` | `"7f0b5d5f7c8696f2d6edcb57e8fa33edb730a9ef084d88189531938105b90a06"` |
| `"records/fit-LGB-b-same_seed_repeat.json"` | `"e1747e0ff395270a91561af1c47e784cc00197500be9f2ab600e94fed79d57ed"` |
| `"records/fit-LGB-b-seed_refit-s43.json"` | `"c9e607a828313be32e0a66ce6c4fb34d181cb83c198b7e0c0ee303cf564d6f86"` |
| `"records/fit-LGB-b-seed_refit-s44.json"` | `"da69d1f9568d654131d22d4bf85cc99bf5b7c3f19eae6103303db34d430a0083"` |
| `"records/fit-LGB-u-same_seed_repeat.json"` | `"e011a68b0e4c6b6bea807b882bf5b5e42568d46f5793d256ed9a9d9370c03295"` |
| `"records/fit-LGB-u-seed_refit-s43.json"` | `"f82579a183112ad78249499475dad460e157f5fd8e8d7d5e40868dc82e18a5d0"` |
| `"records/fit-LGB-u-seed_refit-s44.json"` | `"d2962980f70ca9043cc3de170bf54bf8a08970b38dde9566ac1a77a5df42a2b2"` |
| `"records/fit-RES-b-same_seed_repeat.json"` | `"527a0cc34420e873ea666fd6fe9b67277e33bdb91bf0b29c9e825c7d63f56075"` |
| `"records/fit-RES-b-seed_refit-s43.json"` | `"a944db3a7be956f43a2fc0e00bd3f4573beb494319dba60491e27d22dcb37a67"` |
| `"records/fit-RES-b-seed_refit-s44.json"` | `"79ab05457b2ad09b5db4495213999f80e343278126490cf10c00e818a09fe859"` |
| `"records/fit-RES-u-same_seed_repeat.json"` | `"c68133a5ded03f116949f12a50a24ff033e0b0a65b4cad5dbd9d16835f37c0a1"` |
| `"records/fit-RES-u-seed_refit-s43.json"` | `"9874b797c869a9cec68230ea3145f6793adcf140c06fabe4755ad66e85fe6755"` |
| `"records/fit-RES-u-seed_refit-s44.json"` | `"d193aadb75dac3731417e0e5d2487e05a52768fb0bccc2acbb665644884598f2"` |
| `"records/refit-Cat-b-cpu_refit.json"` | `"0b6c5cbe1d9337e0dbbedb1cd4708497ef24f3170a2ec4ca7b3ab815baa300ef"` |
| `"records/refit-Cat-b-same_seed_repeat.json"` | `"46b33e4c59ce67df4bb316c833386121746b7d39dd4269a3c05d63044049153a"` |
| `"records/refit-Cat-b-seed_refit-s43.json"` | `"fc3109df98c4bf4c2e5c1926d2f6c4aea54d0315bfaf0393858f29c173a6b522"` |
| `"records/refit-Cat-b-seed_refit-s44.json"` | `"5279976939a6525de751c4925b5dcdf5f4bb201fa68077a4687c5bb047ffc24c"` |
| `"records/refit-Cat-u-cpu_refit.json"` | `"8e2a042dbd2e5c201048fb995dfcf562845b00fe1678944f0a645a3312f30ca3"` |
| `"records/refit-Cat-u-same_seed_repeat.json"` | `"535267d0b77e451d6fc0524705cc50aeeca0f11d4fc8e5af6ce1021a2bab723b"` |
| `"records/refit-Cat-u-seed_refit-s43.json"` | `"0e0b145b31219f574725421a36c04135323155b8272cb239ea4f57f3e8a239c7"` |
| `"records/refit-Cat-u-seed_refit-s44.json"` | `"c64c829145a9491aff10dc5384136fa34e717fbbdf10b7ec531f02df5f9f36cc"` |
| `"records/refit-FTT-b-same_seed_repeat.json"` | `"f5520154d502a63b73c31a21576e8aa367f1eba02927265400d2e7a3dfe0170a"` |
| `"records/refit-FTT-b-seed_refit-s43.json"` | `"7e83ca6da0d50bab607768d05ed0b65d4b34ea33e0c0f3a21410c9bd07759dda"` |
| `"records/refit-FTT-b-seed_refit-s44.json"` | `"539e7270ba2c323eba29beefc8f63c5eca2b4939492c43507d58bff0622e4127"` |
| `"records/refit-FTT-u-same_seed_repeat.json"` | `"ec737c5aac1de46c4591b9f350f9a180c9ac3cd452a2cb91f6477b1936498747"` |
| `"records/refit-FTT-u-seed_refit-s43.json"` | `"095d747cc2346e653ce0595ec490aca450667de3957744a043d13ac28aec6179"` |
| `"records/refit-FTT-u-seed_refit-s44.json"` | `"66736990c10472fdb115160d4f313dfd36dc36c08eb2fc860ad575d683fa061e"` |
| `"records/refit-LGB-b-same_seed_repeat.json"` | `"e7d0fe5256e78637b21ee52e291d35886039d6024a4f0c1790aff53b24c878c4"` |
| `"records/refit-LGB-b-seed_refit-s43.json"` | `"93c5faba1271d062f89698ea40c687f443196cccd6eafa9c3edb39bb559b9d97"` |
| `"records/refit-LGB-b-seed_refit-s44.json"` | `"1473e4c9aadc6bc300de97720fd81244c6869e2bddc20a33417d581f09e07020"` |
| `"records/refit-LGB-u-same_seed_repeat.json"` | `"34fcb36863f97c035b222f91c042b3f2f981bc8f00fae86b9224f216f30d9992"` |
| `"records/refit-LGB-u-seed_refit-s43.json"` | `"945abb0c464b4884daed8550522bf641fe11597603a7c1445ea565627cc71594"` |
| `"records/refit-LGB-u-seed_refit-s44.json"` | `"fb32f6e03da027248018bff2e033b9efc42011b26464dbc4f1cc754b27cb4579"` |
| `"records/refit-RES-b-same_seed_repeat.json"` | `"2aa74b075d48e5d8a98b294af33a909a00b31f939980dd9477599327ff88a0d2"` |
| `"records/refit-RES-b-seed_refit-s43.json"` | `"d715cf29390a4b0980e7f00067a771dae6ba57150015ec1d70effc8f777d3fdd"` |
| `"records/refit-RES-b-seed_refit-s44.json"` | `"019b9e1e97d5b8874aec25bd99e9316fdad834bdf703d8d3c90cece364904d45"` |
| `"records/refit-RES-u-same_seed_repeat.json"` | `"c39222ccb62d8d6344b9d8eb4fc37e7b5a19d3c7c30e1c03a3dfefcac51b99b9"` |
| `"records/refit-RES-u-seed_refit-s43.json"` | `"0f1a6ebe33cdfeafe2b979f8b7a65f6f516fb41efe50a8e6a94252a81ef4cbd2"` |
| `"records/refit-RES-u-seed_refit-s44.json"` | `"17c7af3efc40a51f42b131bac82a09642dbaaf1b6d19c849905fae6aa48c3166"` |
| `"records/run.json"` | `"234d1bb57e5f2933062879517ff421d5aa55db81a3be3af3a609963ae1368a9e"` |
| `"records/score-Cat-b-cpu_refit-V2.json"` | `"8045a0eb18401f74c9b314fc84e7ec6c69d6456a8d90da3e868ff057b50c94db"` |
| `"records/score-Cat-b-same_seed_repeat-V2.json"` | `"6ee29843dda7eca71bf14b8230d6e91a5e505be68319b2ab9bc1f80caef44fe3"` |
| `"records/score-Cat-b-seed_refit-s43-V2.json"` | `"209fdd1d5aacc1ea1229110de26984ad0ef4586c175f5b6780588e87f115f5fc"` |
| `"records/score-Cat-b-seed_refit-s44-V2.json"` | `"cfc3fc337ae49edfe68fd7a23fbe7cf4e85cf5ac5381621810fd699df1a1f6fc"` |
| `"records/score-Cat-u-cpu_refit-V2.json"` | `"8ce42219fb29b80860b13ce61a1b0e1313d8c5a87086763242fa5b43f9156be3"` |
| `"records/score-Cat-u-same_seed_repeat-V2.json"` | `"1a9a3415f7d97343ba28ce4f768638ec17e05473242336819881a8132f2d99af"` |
| `"records/score-Cat-u-seed_refit-s43-V2.json"` | `"522d25b74dc8fb4e16391c1e184ce4284e35521e37c1bffa180e4a483dc965ba"` |
| `"records/score-Cat-u-seed_refit-s44-V2.json"` | `"f6767e8aec9b771cc7f0f2dbf1d2ffb403eb9200bf9e1c01b8de85f906002b9c"` |
| `"records/score-FTT-b-same_seed_repeat-V2.json"` | `"b6a3d5673c00f4a686f335a9c1f1352e0814f1d1892e32c38299c67243f03d27"` |
| `"records/score-FTT-b-seed_refit-s43-V2.json"` | `"308092a320b1b8096cb71086e7aa0db5a9a5fc9d9963638d9d4ffd71e1ada8a8"` |
| `"records/score-FTT-b-seed_refit-s44-V2.json"` | `"0df888edd13c14014f9978bed7bef834b41d8e81614fcbb5c9bf71a63796c19b"` |
| `"records/score-FTT-u-same_seed_repeat-V2.json"` | `"fe96eb55d02b6115e79e7b4a447e4b996341d59031a2cfed12cfc8b4288e7332"` |
| `"records/score-FTT-u-seed_refit-s43-V2.json"` | `"b10d5e80dd3fa1e3baa334e439817292ecba6dc9abc25e5192235c87eb53bc20"` |
| `"records/score-FTT-u-seed_refit-s44-V2.json"` | `"4521c4c7746e29af794a88c36ed376e851fcc4a2c3d3084d38aad486e32d6103"` |
| `"records/score-LGB-b-same_seed_repeat-V2.json"` | `"7fe3d04d2c96421bcfd94b61164a66fed588a0244f45ad0670af89acc6292e77"` |
| `"records/score-LGB-b-seed_refit-s43-V2.json"` | `"8a958e7b1aa31e3bfaf9c1a64cae109c10f54c093cbc56c08e4c9cb65b35ada0"` |
| `"records/score-LGB-b-seed_refit-s44-V2.json"` | `"96df74dd133fc30c2314dbe7094030d2f0035cc49604fe02e69462064bc44eaf"` |
| `"records/score-LGB-u-same_seed_repeat-V2.json"` | `"d66fcf663107e4e2b616e1795573e399c784a323ec8792f802ae4e1aa728bc46"` |
| `"records/score-LGB-u-seed_refit-s43-V2.json"` | `"29163846b44103353e020e881b6a9e99e27be338e8b864a7dd3888908ffdeecd"` |
| `"records/score-LGB-u-seed_refit-s44-V2.json"` | `"0c6de8e326b95199ea00e2174e0e50d61a916d7fbaa08660717a280d10e3501f"` |
| `"records/score-RES-b-same_seed_repeat-V2.json"` | `"1bd3f73ace57447940baa252512e545a01f66cf21885e806a83d12e4df000d4a"` |
| `"records/score-RES-b-seed_refit-s43-V2.json"` | `"b82d1a99f64a73cbea6c5566c17dfa2de884fac7a6dc4f978f593d1efa8c4d0e"` |
| `"records/score-RES-b-seed_refit-s44-V2.json"` | `"fe64c23cca877970e5f15699b3a4bfae2c893116359aa42b10953f247ef9acf6"` |
| `"records/score-RES-u-same_seed_repeat-V2.json"` | `"37b6d7e196e986c237e40975ca94312ad25c17d583e3e347e747dd263a0e63c3"` |
| `"records/score-RES-u-seed_refit-s43-V2.json"` | `"7f2eeac675d2f686596ad9c9cfce2534a2e068cab6113235935eaad2bc14d888"` |
| `"records/score-RES-u-seed_refit-s44-V2.json"` | `"6b8472422a3678635242ad3454c3c8a0f3e5c183a9793d61930140d849d709df"` |
